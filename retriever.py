import re
import json
import requests
from py2neo import Graph
import ast
import logging
import os # Added for environment variables if needed
import numpy as np # Added for RAG
import psycopg # Added for RAG
from pgvector.psycopg import register_vector # Added for RAG
from langchain_community.embeddings import HuggingFaceEmbeddings # Added for RAG

# === CONFIGURATION PARAMETERS ===

# --- Data Source Selection ---
# Toggle these variables to control which data sources are included in the LLM context.
# e.g., set USE_PRIVATE_GRAPH_DATA = False to exclude private graph data.
USE_PRIVATE_GRAPH_DATA = True
USE_RAG_DATA = True
USE_UMLS_GRAPH_DATA = True

example_questions = ["{USER PROMPT}"]

# --- Neo4j ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "strongpass123"

# --- Ollama ---
OLLAMA_MODEL = "llama2:70b" # Make sure this model is running via Ollama # Adjusted model name slightly based on common Ollama naming
OLLAMA_URL = "http://localhost:11434/api/generate" # Default Ollama URL

# --- Retrieval Limits ---
MAX_MEDICAL_CONCEPTS = 20     # Maximum number of medical_ns Concept nodes to retrieve per extracted entity (Currently unused, logic focuses on direct/entity match)
MAX_GENERAL_ENTITIES = 20      # Maximum number of general (non-medical_ns Concept) nodes to retrieve per extracted entity (Currently unused)
MAX_RELATIONSHIPS = 20         # Relationships for primary patient search & general entity search
MAX_PARTIAL_NAME_RESULTS = 20 # Limit results per name part to avoid flooding
MAX_GENERAL_ENTITY_RESULTS = 20 # Limit results per general entity search (Adjusted for potentially many entities)

# --- RAG Configuration ---
RAG_CONNECTION_STRING = "postgresql://employees_user:password@localhost:5432/postgres" # Your PGVector connection string
RAG_EMBEDDING_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
RAG_TOP_K = 5 # Number of document chunks to retrieve

try:
    import torch
    if torch.cuda.is_available():
        RAG_DEVICE = "cuda"
    else:
        RAG_DEVICE = "cpu"
except ImportError:
    RAG_DEVICE = "cpu"
    print("Warning: PyTorch not found. Using CPU for embeddings. Install PyTorch for potential GPU acceleration.")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === UTILITY FUNCTIONS ===

# --- DETECT/EXTRACT FUNCTIONS (Unchanged) ---
def detect_patient_name(question):
    """Detects patient names like Mr./Mrs./Ms./Miss Firstname Lastname."""
    pattern = r"(Mr\.|Mrs\.|Ms\.|Miss)\s+([A-Z][a-z]+(?:\s[A-Z][a-z\.]+)*)"
    match = re.search(pattern, question)
    if match:
        return match.group(2)
    return None

def detect_hospital_number(question):
    """Detects hospital numbers (e.g., 40003163, Unknown_76580270) using regex."""
    logging.info('DETECTING HOSPITAL NUMBER')
    pattern = r"\b(\d{8,}|Unknown_\d{8,})\b"
    match = re.search(pattern, question)
    if match:
        return match.group(1)
    return None

def extract_entities_from_question(question):
    """
    Use the LLM to extract a comma-separated list of medical entities,
    including patient names, medical terms, and hospital numbers, from the question.
    Applies stricter prompting and post-processing.
    """
    prompt = f"""
You are an entity extraction engine. Your task is to identify specific entities in the user's question.
Extract the following types of entities if present:
1. Patient Names (e.g., Audrey Chesover, Janet Goodman)
2. Medical Terms or Concepts (e.g., surgery, total hip replacement, pain, SPECT-CT scan, revision arthroplasty)
3. Hospital Numbers

Your output **MUST** be ONLY a comma-separated list containing the extracted entities.
**DO NOT** include any introductory text, explanations, justifications, apologies, or any conversational filler.
Output **ONLY** the comma-separated list.

Question: "{question}"

Extracted entities:
"""
    try:
        logging.info(f"Sending entity extraction request for question: {question}")
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        })
        response.raise_for_status()
        raw_entities_str = response.json().get("response", "").strip()
        logging.info(f'Raw response for Entity Extraction:\n{raw_entities_str}')

        lines = [line.strip() for line in raw_entities_str.split('\n') if line.strip()]
        if not lines:
            logging.warning("LLM returned empty response for entity extraction.")
            return []

        potential_list_str = lines[-1]
        logging.info(f'Using last line for parsing: {potential_list_str}')

        # Basic check if the line looks more like an explanation than a list
        if len(potential_list_str.split()) > 10 and ',' not in potential_list_str :
             logging.warning("Last line looks like explanation, not a list. Attempting to parse anyway, but might be incorrect.")

        # Try splitting by comma first, as requested in the prompt
        entities = [e.strip().strip("'\"") for e in potential_list_str.split(',') if e.strip() and len(e.strip()) > 1]

        # If comma splitting yields very few or no results, try splitting by space as a fallback
        if not entities or len(entities) < 2:
            logging.warning("Comma separation yielded few entities, trying space separation as fallback.")
            space_entities = [e.strip().strip("'\"") for e in re.split(r'\s+', potential_list_str) if e.strip() and len(e.strip()) > 1]
            common_ignore = {"extracted", "entities", "question", "medical", "terms", "concepts", "patient", "names", "hospital", "numbers", "following", "types", "present", "full"}
            space_entities = [e for e in space_entities if e.lower() not in common_ignore]
            entities.extend(se for se in space_entities if se not in entities)

        # Final cleanup
        entities = [e for e in entities if len(e) > 1 and not e.lower().startswith("extracted entities")]

        logging.info(f"LLM Extracted Entities (Processed): {entities}")
        return list(set(entities)) # Return unique entities

    except requests.exceptions.RequestException as e:
        logging.error(f"Error contacting Ollama API for entity extraction: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding Ollama API response for entity extraction: {e}")
        logging.error(f"Raw response text: {response.text}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during entity extraction: {e}")
        return []

# --- GRAPH PROCESSING FUNCTIONS ---
def merge_context(existing_context, new_nodes, new_rels):
    """Merges new nodes/rels into context, avoiding duplicate nodes by ID."""
    node_dict = {}
    # Initialize with existing nodes
    for n in existing_context.get("nodes", []):
        node_key = n.get("id") or n.get("CUI") or n.get("internal_id") or f"neo4j_{n.get('element_id', 'missing_eid')}"
        node_dict[node_key] = n

    # Process new nodes
    for n in new_nodes:
        if "internal_id" not in n:
             n["internal_id"] = n.get("id") or n.get("CUI") or f"neo4j_{n.get('element_id', 'missing_eid')}"

        node_key = n["internal_id"]
        if node_key not in node_dict:
            node_dict[node_key] = n
        else:
            existing_props = node_dict[node_key].setdefault("properties", {})
            new_props = n.get("properties", {})
            for k, v in new_props.items():
                 existing_props.setdefault(k, v)

    existing_context["nodes"] = list(node_dict.values())

    # Process relationships, avoiding duplicates
    existing_rel_tuples = set( (r.get("source"), r.get("target"), r.get("type")) for r in existing_context.get("relationships", []) )
    new_rels_to_add = []
    for rel in new_rels:
        rel_tuple = (rel.get("source"), rel.get("target"), rel.get("type"))
        if rel_tuple not in existing_rel_tuples:
            new_rels_to_add.append(rel)
            existing_rel_tuples.add(rel_tuple)

    existing_context.setdefault("relationships", []).extend(new_rels_to_add)
    return existing_context

def process_cypher_result(records, existing_context, is_patient_record=False):
    """Helper function to process Cypher results and return nodes/rels."""
    current_nodes = []
    current_rels = []
    primary_node_internal_id = None
    processed_node_element_ids = set()

    existing_element_ids = {node.get("element_id") for node in existing_context.get("nodes", []) if node.get("element_id")}

    for record in records:
        p_node = record.get("p") or record.get("n")
        if not p_node:
            continue

        p_node_element_id = p_node.identity
        p_node_props = dict(p_node)
        p_internal_id = p_node_props.get("id") or p_node_props.get("CUI") or f"neo4j_{p_node_element_id}"

        if is_patient_record and primary_node_internal_id is None:
            primary_node_internal_id = p_internal_id

        if p_node_element_id not in existing_element_ids and p_node_element_id not in processed_node_element_ids:
            processed_node_element_ids.add(p_node_element_id)

            node_data = {
                "internal_id": p_internal_id,
                "id": p_node_props.get("id"),
                "CUI": p_node_props.get("CUI"),
                "element_id": p_node_element_id,
                "type": p_node_props.get("type", "Concept" if "CUI" in p_node_props else "Entity"),
                "name": p_node_props.get("name", p_node_props.get("full_name", "Unknown")),
                "properties": {k: v for k, v in p_node_props.items() if k not in ["embedding", "id", "CUI", "name", "type", "full_name"]}
            }
            node_data["properties"]["name"] = node_data["name"]
            node_data["properties"]["type"] = node_data["type"]
            if p_node_props.get("hospital_number"):
                node_data["properties"]["hospital_number"] = p_node_props.get("hospital_number")
            if p_node_props.get("full_name") and "full_name" not in node_data["properties"]:
                node_data["properties"]["full_name"] = p_node_props.get("full_name")

            current_nodes.append(node_data)

        connections = record.get("connections", [])
        for conn in connections:
            n_node = conn.get("target")
            rel_type = conn.get("rel")
            if n_node and rel_type:
                n_node_element_id = n_node.identity
                n_node_props = dict(n_node)
                n_internal_id = n_node_props.get("id") or n_node_props.get("CUI") or f"neo4j_{n_node_element_id}"

                if n_node_element_id not in existing_element_ids and n_node_element_id not in processed_node_element_ids:
                    processed_node_element_ids.add(n_node_element_id)

                    target_node_data = {
                        "internal_id": n_internal_id,
                        "id": n_node_props.get("id"),
                        "CUI": n_node_props.get("CUI"),
                        "element_id": n_node_element_id,
                        "type": n_node_props.get("type", "Concept" if "CUI" in n_node_props else "Entity"),
                        "name": n_node_props.get("name", n_node_props.get("full_name", "Unknown")),
                        "properties": {k: v for k, v in n_node_props.items() if k not in ["embedding", "id", "CUI", "name", "type", "full_name"]}
                    }
                    target_node_data["properties"]["name"] = target_node_data["name"]
                    target_node_data["properties"]["type"] = target_node_data["type"]
                    if n_node_props.get("hospital_number"):
                         target_node_data["properties"]["hospital_number"] = n_node_props.get("hospital_number")
                    if n_node_props.get("full_name") and "full_name" not in target_node_data["properties"]:
                         target_node_data["properties"]["full_name"] = n_node_props.get("full_name")

                    current_nodes.append(target_node_data)

                current_rels.append({
                   "source": p_internal_id,
                   "target": n_internal_id,
                   "type": str(rel_type)
                })

    updated_context = merge_context(existing_context, current_nodes, current_rels)
    return updated_context, primary_node_internal_id

# --- GRAPH RETRIEVAL FUNCTION ---
def retrieve_context_by_cypher(question):
    """Retrieves context from Neo4j using detected Patient Name, or LLM-extracted entities."""
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph.run("RETURN 1") # Test connection
        logging.info("Successfully connected to Neo4j.")
    except Exception as e:
        logging.error(f"Error connecting to Neo4j: {e}")
        raise

    context_info = {"nodes": [], "relationships": []}
    patient_found = False
    primary_patient_internal_id = None
    found_patient_element_ids = set()

    detected_hosp_num = detect_hospital_number(question)
    detected_name_only = detect_patient_name(question)

    # --- Priority 1: Retrieval by Hospital Number ---
    if detected_hosp_num:
        logging.info(f"Detected hospital number: {detected_hosp_num}. Running patient-specific retrieval...")
        cypher_query_hosp = """
            MATCH (p:Entity {hospital_number: $hosp_num})
            WHERE p.type = 'Patient'
            WITH p LIMIT 1
            OPTIONAL MATCH (p)-[r]-(n)
            WITH p, collect({rel: type(r), target: n})[0..$max_rels] AS connections
            RETURN p, connections
        """
        try:
            result_hosp = graph.run(cypher_query_hosp, hosp_num=detected_hosp_num, max_rels=MAX_RELATIONSHIPS).data()
            if result_hosp:
                logging.info(f"Found patient record via hospital number {detected_hosp_num}.")
                context_info, primary_patient_internal_id = process_cypher_result(result_hosp, context_info, is_patient_record=True)
                patient_found = True
                if result_hosp[0]['p']:
                    found_patient_element_ids.add(result_hosp[0]['p'].identity)
            else:
                logging.warning(f"Hospital number {detected_hosp_num} detected, but no matching patient found.")
        except Exception as e:
            logging.error(f"Error querying Neo4j by hospital number: {e}")

    # --- Priority 2: Retrieval by Patient Name (if not found by number) ---
    if not patient_found and detected_name_only:
        logging.info(f"Detected patient name: {detected_name_only}. Running patient-specific retrieval...")
        # 2a: Try exact case-insensitive match first
        cypher_query_name_exact = """
            MATCH (p:Entity)
            WHERE p.type = 'Patient' AND toLower(p.full_name) = toLower($name)
            WITH p LIMIT 1
            OPTIONAL MATCH (p)-[r]-(n)
            WITH p, collect({rel: type(r), target: n})[0..$max_rels] AS connections
            RETURN p, connections
        """
        try:
            result_name_exact = graph.run(cypher_query_name_exact, name=detected_name_only, max_rels=MAX_RELATIONSHIPS).data()
            if result_name_exact:
                logging.info(f"Found patient record via exact name match '{detected_name_only}'.")
                context_info, primary_patient_internal_id = process_cypher_result(result_name_exact, context_info, is_patient_record=True)
                patient_found = True
                if result_name_exact[0]['p']:
                    found_patient_element_ids.add(result_name_exact[0]['p'].identity)
            else:
                # 2b: Try CONTAINS full name (case-insensitive) if exact match fails
                logging.info(f"Exact name match failed for '{detected_name_only}', trying CONTAINS full name...")
                cypher_query_name_contains_full = """
                    MATCH (p:Entity)
                    WHERE p.type = 'Patient' AND toLower(p.full_name) CONTAINS toLower($name)
                    WITH p LIMIT 1
                    OPTIONAL MATCH (p)-[r]-(n)
                    WITH p, collect({rel: type(r), target: n})[0..$max_rels] AS connections
                    RETURN p, connections
                """
                result_name_contains_full = graph.run(cypher_query_name_contains_full, name=detected_name_only, max_rels=MAX_RELATIONSHIPS).data()
                if result_name_contains_full:
                    logging.info(f"Found patient record via CONTAINS full name match '{detected_name_only}'.")
                    context_info, primary_patient_internal_id = process_cypher_result(result_name_contains_full, context_info, is_patient_record=True)
                    patient_found = True
                    if result_name_contains_full[0]['p']:
                        found_patient_element_ids.add(result_name_contains_full[0]['p'].identity)
                else:
                    # 2c: Fallback to CONTAINS for each part of the name
                    logging.info(f"CONTAINS full name match failed for '{detected_name_only}', trying CONTAINS for individual name parts...")
                    name_parts = [part for part in detected_name_only.split() if len(part) > 1]

                    if not name_parts:
                        logging.warning("No valid name parts to search individually.")
                    else:
                        combined_partial_results = []
                        partial_match_element_ids = set()
                        exclude_ids_for_partial = found_patient_element_ids.copy()

                        cypher_query_name_part_contains = """
                            MATCH (p:Entity)
                            WHERE p.type = 'Patient' AND toLower(p.full_name) CONTAINS toLower($name_part)
                            AND NOT elementId(p) IN $exclude_ids
                            WITH p
                            LIMIT $max_partial_results
                            OPTIONAL MATCH (p)-[r]-(n)
                            WITH p, collect({rel: type(r), target: n})[0..$max_rels] AS connections
                            RETURN p, connections
                        """

                        for part in name_parts:
                            logging.info(f"Searching for patients containing name part: '{part}'...")
                            try:
                                partial_result = graph.run(
                                    cypher_query_name_part_contains,
                                    name_part=part,
                                    max_rels=MAX_RELATIONSHIPS,
                                    max_partial_results=MAX_PARTIAL_NAME_RESULTS,
                                    exclude_ids=list(exclude_ids_for_partial)
                                ).data()

                                for record in partial_result:
                                    p_node = record.get("p")
                                    if p_node and p_node.identity not in partial_match_element_ids:
                                        combined_partial_results.append(record)
                                        partial_match_element_ids.add(p_node.identity)
                                        exclude_ids_for_partial.add(p_node.identity)

                            except Exception as e:
                                logging.error(f"Error querying Neo4j by name part '{part}': {e}")

                        if combined_partial_results:
                            logging.info(f"Found {len(combined_partial_results)} potential patient record(s) via partial name matches.")
                            context_info, temp_primary_id = process_cypher_result(combined_partial_results, context_info, is_patient_record=True)
                            if primary_patient_internal_id is None:
                                primary_patient_internal_id = temp_primary_id
                            patient_found = True
                        else:
                             logging.warning(f"Patient name '{detected_name_only}' detected, but no matching patient found via exact, contains full, or contains partial name parts.")

        except Exception as e:
            logging.error(f"Error querying Neo4j by patient name: {e}")

    # --- Priority 3: General entity retrieval (Condition logic remains unchanged, only affects fallback) ---
    try:
        entities = extract_entities_from_question(question)
        if detected_name_only and detected_name_only in entities:
                entities.remove(detected_name_only)
                logging.info(f"Removing failed name '{detected_name_only}' from general entity search list.")
        if detected_hosp_num and detected_hosp_num in entities:
                entities.remove(detected_hosp_num)
                logging.info(f"Removing failed hosp number '{detected_hosp_num}' from general entity search list.")
        logging.info(f"Entities for general search: {entities}")
    except Exception as e:
        logging.error(f"Failed to extract entities using LLM: {e}")
        entities = []

    if entities:
        logging.info("Performing general entity search across relevant properties...")
        combined_general_results = []
        general_entity_element_ids = set()
        exclude_ids_for_general = {node.get("element_id") for node in context_info.get("nodes", []) if node.get("element_id")}

        cypher_query_general_entity = """
            MATCH (n)
            WHERE NOT elementId(n) in $exclude_ids
            WITH n
            WHERE (
                (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($entity_lower)) OR
                (n.full_name IS NOT NULL AND toLower(n.full_name) CONTAINS toLower($entity_lower)) OR
                (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($entity_lower)) OR
                (n.type IS NOT NULL AND toLower(n.type) CONTAINS toLower($entity_lower))
            )
            WITH n
            LIMIT $max_results
            OPTIONAL MATCH (n)-[r]-(m)
            WITH n, collect({rel: type(r), target: m})[0..$max_general_rels] AS connections
            RETURN n, connections
        """

        for entity in entities:
            entity_lower = entity.lower()
            logging.info(f" --> Searching generally for entity: '{entity}'")
            try:
                result_entity = graph.run(
                    cypher_query_general_entity,
                    entity_lower=entity_lower,
                    max_results=MAX_GENERAL_ENTITY_RESULTS,
                    max_general_rels=MAX_RELATIONSHIPS,
                    exclude_ids=list(exclude_ids_for_general)
                ).data()

                for record in result_entity:
                    n_node = record.get("n")
                    if n_node and n_node.identity not in general_entity_element_ids:
                        combined_general_results.append(record)
                        general_entity_element_ids.add(n_node.identity)
                        exclude_ids_for_general.add(n_node.identity)

            except Exception as e:
                logging.error(f"Error querying Neo4j for general entity '{entity}': {e}")

        if combined_general_results:
            logging.info(f"Processing {len(combined_general_results)} total potential general context nodes found across all entities.")
            context_info, _ = process_cypher_result(combined_general_results, context_info, is_patient_record=False)
        else:
            logging.info("No general context found matching the extracted entities.")
    else:
        logging.warning("No entities extracted or remaining for general search.")

    num_nodes = len(context_info.get("nodes", []))
    num_rels = len(context_info.get("relationships", []))
    logging.info(f"Graph retrieval finished. Raw context contains {num_nodes} nodes and {num_rels} relationships (filtering applied during formatting).")

    return context_info


# --- RAG FUNCTIONS (Unchanged) ---

_embedding_model = None # Cache the embedding model

def get_embedding_model():
    """Initialize and cache the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        logging.info(f"Initializing embedding model: {RAG_EMBEDDING_MODEL_NAME} on device: {RAG_DEVICE}")
        try:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=RAG_EMBEDDING_MODEL_NAME,
                model_kwargs={"device": RAG_DEVICE},
                encode_kwargs={"device": RAG_DEVICE, "batch_size": 100},
            )
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
    return _embedding_model

def semantic_search(query_text, k=RAG_TOP_K, detected_hosp_num=None):
    """
    Perform semantic search in the vector database for the given query text.
    """
    logging.info(f"Performing RAG search. Semantic top K={k}. Hospital number provided: {'Yes (' + detected_hosp_num + ')' if detected_hosp_num else 'No'}")
    if not query_text:
        logging.warning("Semantic search query text is empty. Skipping RAG.")
        return []

    final_results = []
    processed_ids = set()

    try:
        embedding_model = get_embedding_model()
        logging.info(f"Connecting to vector database: {RAG_CONNECTION_STRING.split('@')[-1]}")
        with psycopg.connect(RAG_CONNECTION_STRING, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:

                # Step 1 - Direct retrieval by Hospital Number
                if detected_hosp_num:
                    direct_match_query = "SELECT id, text, source FROM documents WHERE source LIKE %s"
                    try:
                        cur.execute(direct_match_query, ('%' + detected_hosp_num + '%',))
                        direct_rows = cur.fetchall()
                        logging.info(f"Found {len(direct_rows)} chunks via direct hospital number match.")
                        for row in direct_rows:
                            chunk_id = row[0]
                            final_results.append({"id": chunk_id, "text": row[1], "source": row[2]})
                            processed_ids.add(chunk_id) # Keep for consistency

                        logging.info(f"RAG retrieval finished (Hospital Number Match Only). Returning {len(final_results)} chunks.")
                        return final_results
                    except Exception as e:
                        logging.error(f"Error during direct hospital number retrieval: {e}")

                # Step 2 - Semantic Search
                logging.info("Generating embedding for the query...")
                embedding = np.array(embedding_model.embed_query(query_text))
                logging.info("Query embedding generated. Performing semantic search...")

                cur.execute(
                    "SELECT id, text, source FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
                    (embedding, k),
                )
                semantic_rows = cur.fetchall()
                logging.info(f"Retrieved {len(semantic_rows)} chunks via semantic search (before filtering).")

                # Step 3 - Combine and Deduplicate
                for row in semantic_rows:
                    chunk_id = row[0]
                    if chunk_id not in processed_ids:
                        final_results.append({"id": chunk_id, "text": row[1], "source": row[2]})
                        processed_ids.add(chunk_id)

        logging.info(f"RAG retrieval finished. Returning {len(final_results)} unique chunks.")
        return final_results

    except psycopg.OperationalError as e:
        logging.error(f"Database connection error during RAG search: {e}")
        return []
    except Exception as e:
        logging.error(f"An error occurred during RAG search: {e}")
        return []

def format_graph_context_for_prompt(context, use_private, use_umls):
    """
    Formats the graph subgraph data into a readable string for the LLM,
    filtering based on configuration flags (use_private, use_umls).
    - For Concept nodes: Includes only Name, Type, and Definition (if use_umls is True).
    - For other nodes: Includes Name, Type, Other Details, and Relationships
                      (if use_private is True). Relationships are filtered if their
                      target node type is disabled.
    """
    nodes = context.get("nodes", [])
    relationships = context.get("relationships", [])

    if not nodes:
        return "No relevant information was retrieved from the knowledge graph."
    if not use_private and not use_umls:
        return "Knowledge graph context disabled by configuration."

    node_lookup = {n["internal_id"]: n for n in nodes if "internal_id" in n}

    # --- Relationship Grouping (Modified to filter based on flags) ---
    node_relationships = {}
    processed_rel_tuples = set()
    for rel in relationships:
        source_id = rel.get("source")
        target_id = rel.get("target")
        rel_type = rel.get("type")

        if not all([source_id, target_id, rel_type, source_id in node_lookup, target_id in node_lookup]):
            continue

        source_node = node_lookup[source_id]
        target_node = node_lookup[target_id]

        # Determine type of source and target nodes
        source_is_concept = source_node.get('CUI') or source_node.get('type') == 'Concept'
        target_is_concept = target_node.get('CUI') or target_node.get('type') == 'Concept'

        # Apply filtering based on configuration flags
        if source_is_concept and not use_umls:
            # logging.debug(f"Skipping rel: Source '{source_node.get('name')}' is Concept, but USE_UMLS_GRAPH_DATA is False.")
            continue
        if not source_is_concept and not use_private:
            # logging.debug(f"Skipping rel: Source '{source_node.get('name')}' is Entity, but USE_PRIVATE_GRAPH_DATA is False.")
            continue
        if target_is_concept and not use_umls:
            # logging.debug(f"Skipping rel: Target '{target_node.get('name')}' is Concept, but USE_UMLS_GRAPH_DATA is False.")
            continue
        if not target_is_concept and not use_private:
            # logging.debug(f"Skipping rel: Target '{target_node.get('name')}' is Entity, but USE_PRIVATE_GRAPH_DATA is False.")
            continue


        if source_node and target_node: # Add a check to ensure nodes exist before printing names
            processed_rel_type = str(rel_type).strip().lower()

        # Filter out 'reference of' relationship type explicitly as before
        if str(rel_type).strip().lower() == 'reference_of':
            continue

        rel_tuple_key = tuple(sorted((str(source_id), str(target_id)))) + (str(rel_type),)
        if rel_tuple_key in processed_rel_tuples:
            continue

        # Get names for formatting
        source_name = (source_node.get('name') or
                       source_node.get('properties', {}).get('full_name') or
                       source_node.get('properties', {}).get('name') or
                       f"Node_{source_id}")
        target_name = (target_node.get('name') or
                       target_node.get('properties', {}).get('full_name') or
                       target_node.get('properties', {}).get('name') or
                       target_node.get('properties', {}).get('description') or
                       f"Node_{target_id}")

        rel_type_fmt = str(rel_type).replace("_", " ").lower()
        relationship_desc = f"`{source_name}` --[{rel_type_fmt}]--> `{target_name}`"

        # Store the formatted description string, keyed by source node ID
        node_relationships.setdefault(source_id, []).append(relationship_desc)
        processed_rel_tuples.add(rel_tuple_key)

    context_parts = []
    processed_node_ids = set()
    included_node_count = 0

    for node_id, node in node_lookup.items():
        if node_id in processed_node_ids:
             continue

        node_type = node.get('type', 'Entity')
        name = node.get('name', f'Unknown_{node_type}')
        props = node.get('properties', {})
        is_concept = node.get('CUI') or node_type == 'Concept'

        # Apply filtering based on node type and flags
        if is_concept and not use_umls:
            # logging.debug(f"Skipping node '{name}': Is Concept, but USE_UMLS_GRAPH_DATA is False.")
            continue
        if not is_concept and not use_private:
            # logging.debug(f"Skipping node '{name}': Is Entity, but USE_PRIVATE_GRAPH_DATA is False.")
            continue

        node_desc = ""

        if is_concept:
            # Format Concept nodes (UMLS)
            node_desc = f"- **{name}** (Type: `{node_type}`)"
            definition = props.get('definition')
            if definition:
                definition_str = str(definition).strip()
                if len(definition_str) > 250: # Truncate
                    definition_str = definition_str[:247] + "..."
                node_desc += f"\n  Definition: {definition_str}"
        else:
            # Format Entity nodes (Private)
            node_desc = f"- **{name}** (Type: `{node_type}`)"

            prop_strings = []
            sorted_props = sorted(props.items())
            excluded_props = {"id", "name", "type", "embedding", "namespace", "CUI",
                              "element_id", "internal_id", "full_name", "definition"}
            if node.get("name") == props.get("full_name"):
                 excluded_props.add("full_name")

            for prop, value in sorted_props:
                if prop not in excluded_props and value not in [None, ""]:
                    prop_val_str = str(value)
                    if len(prop_val_str) > 150:
                        prop_val_str = prop_val_str[:147] + "..."
                    prop_key_fmt = prop.replace('_', ' ').capitalize()
                    prop_strings.append(f"{prop_key_fmt}: `{prop_val_str}`")

            if prop_strings:
                node_desc += f"\n  Other Details: {'; '.join(prop_strings)}"

            # Add filtered relationships (already filtered in the grouping phase)
            if node_id in node_relationships and node_relationships[node_id]:
                sorted_filtered_rels = sorted(node_relationships[node_id]) # Sort remaining rels
                if sorted_filtered_rels:
                    rels_str = '\n  Relationships:\n    - ' + '\n    - '.join(sorted_filtered_rels)
                    node_desc += rels_str

        # Append the formatted description to the list if it's not empty
        if node_desc.strip():
            context_parts.append(node_desc)
            included_node_count += 1
        processed_node_ids.add(node_id) # Mark node as processed

    logging.info(f"Formatted {included_node_count} graph nodes based on configuration.")

    # Join parts, filtering potentially empty entries
    final_context_str = "\n\n".join(filter(str.strip, context_parts))

    if not final_context_str.strip():
        # Adjust message based on whether filtering occurred
        if not use_private and not use_umls:
            return "Knowledge graph context disabled by configuration."
        else:
            return "No specific information could be formatted from the knowledge graph based on the selected sources."

    return final_context_str


def format_rag_context_for_prompt(rag_results, use_rag_flag):
    """Formats the RAG results into a readable string for the LLM."""
    if not use_rag_flag:
        return "Document context (RAG) disabled by configuration."
    if not rag_results:
        return "No relevant information was retrieved from document sources."

    context_parts = []
    for i, chunk in enumerate(rag_results, 1):
        original_source = chunk.get('source', 'Unknown Source')
        text = chunk.get('text', 'No text available')
        hospital_number = None
        match = re.search(r'(\d{8})', original_source)
        if match:
            hospital_number = match.group(1)
            source_display = f"Hospital No. {hospital_number}"
        else:
            source_display = f"Source: {original_source} (Unknown Hospital No.)"
            logging.warning(f"Could not extract 8-digit hospital number from source: {original_source}")

        context_parts.append(f"Chunk {i} ({source_display}):\n{text}\n---")

    if context_parts:
        context_parts[-1] = context_parts[-1].rstrip('---\n')

    return "\n".join(context_parts)


# --- LLM GENERATION FUNCTION ---
def generate_response_with_context(question, combined_context_str):
    """Generates response using LLM with the combined context."""
    prompt = f"""
You are a medical information assistant.
Your task is to answer the user's question based *only* on the provided context below.
The context includes information retrieved from selected sources (knowledge graph entities/concepts, document chunks).
Synthesize the information from the available sources, but do not add information not present in the context.
If the context does not contain the answer, or if the necessary sources are disabled, state that the information is not available in the provided context or based on the current configuration.

**Combined Context:**
{combined_context_str}

**User Question:**
{question}

**Answer based strictly on the combined context:**
"""

    logging.info("--- Sending Prompt to LLM ---")
    log_prompt = prompt[:1000] + "..." if len(prompt) > 1000 else prompt
    # logging.debug(log_prompt)
    logging.info("--- End of Prompt ---")

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        })
        response.raise_for_status()
        llm_response = response.json().get("response", "").strip()
        logging.info("LLM generated response successfully.")
        # logging.debug(f"LLM Raw Response: {llm_response}")
        return llm_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error contacting Ollama API during generation: {e}")
        return f"Error: Could not generate response due to API connection issue. {e}"
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding Ollama API response during generation: {e}")
        logging.error(f"Raw response text: {response.text}")
        return f"Error: Could not generate response due to API decoding issue. {e}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM generation: {e}")
        return f"Error: An unexpected issue occurred while generating the response. {e}"

# --- MAIN RETRIEVER FUNCTION ---

def hybrid_retriever(question):
    """
    Retrieves context from potentially Neo4j graph and Vector DB (RAG), based on global flags,
    then generates an answer using an LLM.
    """
    logging.info(f"Processing question: {question}")
    logging.info(f"Data Source Config: PrivateGraph={USE_PRIVATE_GRAPH_DATA}, UMLSGraph={USE_UMLS_GRAPH_DATA}, RAG={USE_RAG_DATA}")

    # 1. Retrieve Graph Context (Always retrieve raw data)
    graph_context_data = {"nodes": [], "relationships": []} # Default empty
    # Only retrieve if at least one graph source is enabled
    if USE_PRIVATE_GRAPH_DATA or USE_UMLS_GRAPH_DATA:
        logging.info("STEP 1: Retrieving context from Neo4j Knowledge Graph (Sources Enabled)...")
        try:
            graph_context_data = retrieve_context_by_cypher(question)
        except Exception as e:
            logging.error(f"Failed during graph context retrieval: {e}")
    else:
        logging.info("STEP 1: Skipping Neo4j Knowledge Graph retrieval (All Graph Sources Disabled).")


    # 2. Format Graph Context for LLM (Applying filters based on flags)
    logging.info("STEP 2: Formatting graph context (applying source filters)...")
    # <<< MODIFICATION START: Pass flags to formatting function >>>
    graph_context_str_formatted = format_graph_context_for_prompt(
        graph_context_data,
        USE_PRIVATE_GRAPH_DATA,
        USE_UMLS_GRAPH_DATA
    )
    print(f"\n--- Formatted Graph Context (Filtered) ---\n{graph_context_str_formatted}\n------------------------------")


    # 3. Perform Semantic Search (RAG) - Conditional execution
    rag_results = []
    rag_context_str_formatted = "" # Initialize
    detected_hosp_num_for_rag = detect_hospital_number(question) # Detect always, use if RAG enabled

    if USE_RAG_DATA:
        logging.info("STEP 3: Performing semantic search (RAG) on Vector DB (Enabled)...")
        # Construct the query for RAG
        node_names = [n.get('name', '') for n in graph_context_data.get('nodes', [])] # Use raw data before filtering here
        rag_query_input = f"{question}\n\nKey concepts from graph: {', '.join(filter(None, node_names))}"
        logging.info(f"Using RAG query input (first 500 chars): {rag_query_input[:500]}...")
        try:
            rag_results = semantic_search(
                query_text=rag_query_input,
                k=RAG_TOP_K,
                detected_hosp_num=detected_hosp_num_for_rag
            )
        except Exception as e:
            logging.error(f"Failed during semantic search (RAG): {e}")
            rag_results = [] # Reset on error

        # 4. Format RAG Context for LLM - Conditional execution
        logging.info("STEP 4: Formatting RAG context (Enabled)...")
        rag_context_str_formatted = format_rag_context_for_prompt(rag_results, USE_RAG_DATA) # Pass flag

    else:
        logging.info("STEP 3 & 4: RAG data retrieval and formatting skipped (Disabled by configuration).")
        # Ensure the placeholder message is set if RAG is disabled
        rag_context_str_formatted = format_rag_context_for_prompt([], USE_RAG_DATA) # Pass flag to get disabled message

    # 5. Combine Contexts for the Final LLM Prompt (Respecting flags)
    logging.info("STEP 5: Combining contexts for LLM based on configuration...")
    combined_context_parts = []

    # Add Graph section only if it wasn't disabled AND produced some output
    if USE_PRIVATE_GRAPH_DATA or USE_UMLS_GRAPH_DATA:
         combined_context_parts.append(f"**Retrieved Medical Knowledge Graph Context:**\n{graph_context_str_formatted}")
    else:
         # Explicitly state graph was disabled if both flags are false
         combined_context_parts.append("**Knowledge Graph Context:** Disabled by configuration.")

    # Add RAG section (format_rag_context_for_prompt handles disabled message internally now)
    combined_context_parts.append(f"**Retrieved Relevant Document Chunks:**\n{rag_context_str_formatted}")


    combined_context = "\n\n".join(combined_context_parts).strip()

    # Handle case where ALL sources are disabled or yield no info
    if not (USE_PRIVATE_GRAPH_DATA or USE_UMLS_GRAPH_DATA or USE_RAG_DATA):
         combined_context = "All data sources (Graph, RAG) are disabled by configuration. Cannot answer question based on provided context."
    elif not graph_context_str_formatted.strip().startswith("No specific information") and \
         not graph_context_str_formatted.strip().startswith("Knowledge graph context disabled") and \
         not rag_context_str_formatted.strip().startswith("No relevant information") and \
         not rag_context_str_formatted.strip().startswith("Document context (RAG) disabled"):
         # If all sources enabled but yielded *nothing*, add a note.
         # This check is a bit complex, might need refinement based on exact empty messages.
         pass


    # logging.info("Combined context prepared.")
    print("\n=== Combined Context for LLM (Filtered) ===")
    print(combined_context)
    print("=== End of Combined Context ===\n")

    # 6. Generate Final Answer using LLM
    logging.info("STEP 6: Generating final answer using LLM...")
    final_answer = generate_response_with_context(question, combined_context)

    return final_answer

# === MAIN WORKFLOW ===
if __name__ == "__main__":
    try:
       get_embedding_model()
    except Exception as e:
       print(f"Could not pre-load embedding model: {e}. Will attempt loading on first query.")

    for q in example_questions:
        print("\n===============================================")
        print(f"Question: {q}\n")
        try:
            # Call the retriever which now respects the global flags
            final_answer = hybrid_retriever(q)
            print("\n--- Final Answer ---")
            print(final_answer)
            print("--- End Answer ---")
        except Exception as e:
            print(f"\n==== Error Processing Question ==== ")
            print(f"Question: {q}")
            print(f"An error occurred during the process: {e}")
            import traceback
            traceback.print_exc()
        print("===============================================\n")
