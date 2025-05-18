import os
import json
import re
import requests
import pdfplumber
from neo4j import GraphDatabase
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Union
import uuid
import logging
from sentence_transformers import SentenceTransformer, models

# --- Create the model with explicit pooling ---
# Load the transformer component
word_embedding_model = models.Transformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# Create a pooling layer that uses the mean of token embeddings only.
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), 
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
# Build the SentenceTransformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Entity types to exclude from embedding
EXCLUDED_TYPES_FOR_EMBEDDING = {"Patient", "Clinician", "GP", "Hospital", "Visit"}

# === CONFIGURATION ===
PDF_DIR = "../datasets/dataset/"
OLLAMA_MODEL = "llama3.3:70b"
OLLAMA_URL = "http://localhost:11434/api/generate"
NEO4J_URI = "bolt://localhost:7690"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
MAX_RETRIES = 3

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === SETUP NEO4J CLIENT ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === Pydantic Models for Validation ===
class Entity(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    properties: Dict[str, Union[str, List[str], List[float], None]]

class Relationship(BaseModel):
    source: str
    target: str
    type: str

class GraphData(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# === UTILITIES ===
def extract_text_from_pdf(file_path):
    text = ""
    images = convert_from_path(file_path, dpi=300)
    for i, img in enumerate(images):
        logger.info(f"Running OCR on page {i + 1} of {file_path}")
        page_text = pytesseract.image_to_string(img)
        if page_text:
            text += page_text + "\n"
    logger.info(f"Extracted text from {file_path}: {text[:500]}...")
    if not text.strip():
        logger.warning(f"No text extracted from {file_path}")
    return text

def ask_ollama_for_graph_structure(text, hospital_number, retry_count=0):
    prompt = f"""
You are an expert medical information extraction system.

From the following clinical document, extract all medically relevant **entities** and **relationships** to build a knowledge graph for diagnostic reasoning, treatment tracking, patient history analysis, and research.

### Instructions:
1. **Entities**: Extract with all available properties under a `properties` key. **Do not assign an `id` field**; it will be assigned programmatically. **Always include a `name` property** for every entity, derived from the most relevant field (e.g., `full_name` for Patient, `condition_name` for Diagnosis, `description` for Symptom, etc.). **Ensure at least one Patient entity is extracted per document** by looking for any mention of a patient's name, even if other details are missing (use hospital_number as a fallback identifier). Include:
   - Patient: full_name, date_of_birth, gender, nhs_number, address, occupation, medical_history, comorbidities, allergies, lifestyle_factors, medications, hospital_number (set `name` to `full_name` or "Unknown Patient {hospital_number}" if full_name is missing, **set `hospital_number` to "{hospital_number}" if not found in the document**)
   - Clinician: full_name, title, role, specialty, hospital_affiliation, contact_info (set `name` to `full_name`)
   - GP / Referring Doctor: name, practice, address (set `name` to `name`)
   - Hospital / Clinic / Trust: name, address, department, contact_info (set `name` to `name`)
   - Diagnosis: condition_name, side, severity, chronicity, date_of_diagnosis, underlying_cause (set `name` to `condition_name`)
   - Symptom: description, location, severity, duration, onset (set `name` to `description`)
   - Procedure / Surgery: name, side, date, outcome, reason_for_procedure, complications (set `name` to `name`)
   - Imaging / Tests: modality, date, findings, result_summary (set `name` to `modality`)
   - Medication: name, dosage, frequency, indication, duration, side_effects (set `name` to `name`)
   - Referral: referring_clinician, referred_to, reason (set `name` to `referring_clinician`)
   - Injury / Event: type, date, context, resulting_condition (set `name` to `type`)
   - Visit / Appointment: date, reason, outcome (set `name` to `date`)
   - Body Part / Anatomical Site: location (set `name` to `location`)
   - Comorbidity: condition (set `name` to `condition`)
   - Allergy: substance, reaction, severity (set `name` to `substance`)

2. **Relationships**: Specify `source` (index of the source entity in the entities list), `target` (index of the target entity in the entities list), and `type` as strings. Use the exact keys `source`, `target`, and `type`, and use the following relationship types:
   - has_diagnosis (Patient to Diagnosis)
   - experiences_symptom (Patient to Symptom)
   - suggests_diagnosis (Symptom to Diagnosis)
   - treats_diagnosis (Procedure or Medication to Diagnosis)
   - underwent_procedure (Patient to Procedure)
   - performed_procedure (Clinician to Procedure)
   - had_imaging (Patient to Imaging)
   - confirmed_diagnosis (Imaging to Diagnosis)
   - prescribed_medication (Clinician to Medication)
   - administered_medication (Patient to Medication)
   - treated_by (Patient to Clinician)
   - treated_at (Patient to Hospital)
   - affiliated_with (Clinician to Hospital)
   - referred_to (GP or Clinician to another Clinician/Department)
   - follow_up (Visit to future plan)
   - has_comorbidity (Patient to Comorbidity)
   - has_allergy (Patient to Allergy)
   - caused_by_injury (Diagnosis to Injury)
   - examined_by (Patient to Clinician)
   - has_body_site (Diagnosis/Symptom/Procedure to Anatomical Site)

### Output Format:
Return **only** a valid JSON object with `entities` and `relationships`. Each entity must have `type` and `properties` (including a `name` property). Do not include an `id` field in entities. Relationships must have `source`, `target` (as indices of entities in the entities list), and `type`. Do not include any additional text, explanations, or comments outside the JSON.

### Error Handling:
- If you cannot extract any entities or relationships, include at least one Patient entity with `full_name: "Unknown Patient {hospital_number}"` and `hospital_number: "{hospital_number}"`.
- Ensure all fields are properly formatted as strings or lists of strings in the `properties` dictionary.
- Do not include any invalid JSON characters or unescaped quotes.
- Use the exact keys `source`, `target`, and `type` for relationships.

Document:
{text[:15000]}
"""

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code != 200:
        logger.error(f"Ollama API Error: {response.status_code} - {response.text}")
        return GraphData(entities=[], relationships=[])

    result = response.json().get("response", "")
    logger.info(f"Raw Ollama Response (Retry {retry_count + 1}): {result[:1000]}...")

    # Try to extract JSON from the response
    sanitized = re.search(r"\{[\s\S]*\}", result)
    if not sanitized:
        logger.error(f"No valid JSON found in response: {result}")
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying ({retry_count + 1}/{MAX_RETRIES})...")
            return ask_ollama_for_graph_structure(text, hospital_number, retry_count + 1)
        logger.error("Max retries reached. Returning fallback patient.")
        return GraphData(
            entities=[Entity(
                id=str(uuid.uuid4()),
                type="Patient",
                properties={"name": f"Unknown Patient {hospital_number}", "hospital_number": hospital_number}
            )],
            relationships=[]
        )

    try:
        raw_data = json.loads(sanitized.group(0))
        # Assign unique IDs to entities and ensure hospital_number for patients
        entities_with_ids = []
        for entity_data in raw_data.get("entities", []):
            properties = entity_data["properties"]
            entity_type = entity_data["type"]

            # Ensure hospital_number is set for Patient entities
            if entity_type == "Patient":
                if "hospital_number" not in properties or properties["hospital_number"] is None:
                    logger.warning(f"Patient entity missing hospital_number. Setting to {hospital_number}.")
                    properties["hospital_number"] = hospital_number

            # Generate embedding for allowed types
            if entity_type not in EXCLUDED_TYPES_FOR_EMBEDDING:
                # Embed based on available string values in the properties
                combined_text = " ".join(str(val) for val in properties.values() if isinstance(val, str))
                if combined_text.strip():  # Only embed if there's content
                    # Generate a 768-dimensional embedding
                    embedding = model.encode(combined_text, normalize_embeddings=True).tolist()
                    properties["embedding"] = embedding

            entities_with_ids.append(Entity(
                id=str(uuid.uuid4()),
                type=entity_type,
                properties=properties
            ))

        # Transform relationships from indices to IDs
        transformed_relationships = []
        for rel in raw_data.get("relationships", []):
            source_idx = rel["source"]
            target_idx = rel["target"]
            if 0 <= source_idx < len(entities_with_ids) and 0 <= target_idx < len(entities_with_ids):
                transformed_relationships.append(Relationship(
                    source=entities_with_ids[source_idx].id,
                    target=entities_with_ids[target_idx].id,
                    type=rel["type"]
                ))
            else:
                logger.warning(f"Invalid relationship indices: source={source_idx}, target={target_idx}, skipping.")

        graph_data = GraphData(entities=entities_with_ids, relationships=transformed_relationships)

        # Ensure at least one patient exists
        patient_entities = [e for e in graph_data.entities if e.type == "Patient"]
        if not patient_entities:
            logger.warning(f"No Patient entity found in {hospital_number}. Adding fallback patient.")
            graph_data.entities.append(Entity(
                id=str(uuid.uuid4()),
                type="Patient",
                properties={"name": f"Unknown Patient {hospital_number}", "hospital_number": hospital_number}
            ))

        # Log extracted objects
        logger.info(f"Extracted {len(graph_data.entities)} entities and {len(graph_data.relationships)} relationships for hospital_number {hospital_number}")
        for entity in graph_data.entities:
            logger.info(f"Entity: {entity.type} - ID: {entity.id} - Properties: {entity.properties}")
        for rel in graph_data.relationships:
            logger.info(f"Relationship: {rel.source} -> {rel.target} ({rel.type})")

        return graph_data
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Parsing error: {e}")
        logger.debug(f"Raw response: {sanitized.group(0)}")
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying ({retry_count + 1}/{MAX_RETRIES})...")
            return ask_ollama_for_graph_structure(text, hospital_number, retry_count + 1)
        logger.error("Max retries reached. Returning fallback patient.")
        return GraphData(
            entities=[Entity(
                id=str(uuid.uuid4()),
                type="Patient",
                properties={"name": f"Unknown Patient {hospital_number}", "hospital_number": hospital_number}
            )],
            relationships=[]
        )

def insert_graph_into_neo4j(graph_data, namespace="default_ns"):
    if not graph_data or not graph_data.entities:
        logger.warning("No valid graph data to insert.")
        return

    with driver.session() as session:
        for entity in graph_data.entities:
            entity.properties["namespace"] = namespace  # ✅ Add namespace to properties

            if entity.type == "Patient":
                hospital_number = entity.properties.get("hospital_number")
                if hospital_number is None:
                    logger.error(f"Patient entity missing hospital_number after validation: {entity.properties}")
                    raise ValueError("Patient entity must have a hospital_number")
                session.run(
                    """
                    MERGE (e:Entity {hospital_number: $hospital_number})
                    SET e.id = $id, e += $props, e.type = $type, e.name = $name
                    """,
                    hospital_number=hospital_number,
                    id=entity.id,
                    props=entity.properties,
                    type=entity.type,
                    name=entity.properties.get("name", f"Unknown {entity.type}")
                )
            else:
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e += $props, e.type = $type, e.name = $name
                    """,
                    id=entity.id,
                    props=entity.properties,
                    type=entity.type,
                    name=entity.properties.get("name", f"Unknown {entity.type}")
                )
            logger.info(f"Inserted entity: {entity.type} - {entity.properties.get('name', 'Unknown')} (ID: {entity.id})")

        for rel in graph_data.relationships:
            session.run(
                f"""
                MATCH (a:Entity {{id: $source}})
                MATCH (b:Entity {{id: $target}})
                MERGE (a)-[r:{rel.type}]->(b)
                """,
                source=rel.source,
                target=rel.target
            )
            logger.info(f"Inserted relationship: {rel.source} -> {rel.target} ({rel.type})")

def process_all_pdfs():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    total_patients = 0

    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        path = os.path.join(PDF_DIR, filename)
        match = re.search(r"\d{4}-\d{2}-\d{2}\s(\d+)\s", filename)
        hospital_number = match.group(1) if match else f"Unknown_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Processing file: {filename}, Hospital Number: {hospital_number}")
        text = extract_text_from_pdf(path)
        graph_data = ask_ollama_for_graph_structure(text, hospital_number)

        patients_in_file = sum(1 for entity in graph_data.entities if entity.type == "Patient")
        total_patients += patients_in_file
        logger.info(f"Found {patients_in_file} patients in {filename}. Total patients so far: {total_patients}")

        insert_graph_into_neo4j(graph_data)

    logger.info(f"Finished processing. Total patients found: {total_patients}")

if __name__ == "__main__":
    process_all_pdfs()
