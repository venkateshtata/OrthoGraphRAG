import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models

tqdm.pandas()

# === Load SRDEF.RRF for semantic type names and definitions ===
df_srdef = pd.read_csv('/YOUR_FILE_LOCATION/SRDEF', sep='|', header=None, dtype=str)
df_srdef.columns = ['Type', 'TUI', 'STN', 'Hierarchy', 'STY_DEF', 'Na1', 'Na2', 'Na3', 'Abbr', 'Na4', 'Na5']
df_srdef = df_srdef[df_srdef['Type'] == 'STY']

tui_to_stn = df_srdef.set_index('TUI')['STN'].to_dict()
tui_to_def = df_srdef.set_index('TUI')['STY_DEF'].to_dict()

# === Load MRCONSO.RRF ===
df_conso = pd.read_csv('/YOUR_FILE_LOCATION/MRCONSO.RRF.aa.gz', sep='|', header=None, dtype=str)
df_conso.columns = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
    "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF", "EMPTY"
]
df_conso = df_conso[(df_conso['LAT'] == 'ENG') & (df_conso['TTY'] == 'PT')]

# === Load MRSTY: Concept Semantic Types ===
df_sty = pd.read_csv('/YOUR_FILE_LOCATION/MRSTY.RRF.gz', sep='|', header=None, dtype=str)
df_sty = df_sty.iloc[:, :6]
df_sty.columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]

# Inject STN and Definition from SRDEF
df_sty['STN'] = df_sty['TUI'].map(tui_to_stn)
df_sty['SEMDEF'] = df_sty['TUI'].map(tui_to_def)

# === Filter: Orthopaedic-relevant types ===
ortho_keywords = [
    # Anatomy & Pathology
    "Musculoskeletal System",
    "Bone",
    "Joint",
    "Cartilage",
    "Ligament",
    "Tendon",
    "Muscle",
    "Spine",
    "Hip",
    "Knee",
    "Shoulder",
    "Ankle",
    "Wrist",
    "Hand",
    "Foot",
    "Pelvis",
    "Femur",
    "Tibia",
    "Fibula",
    "Patella",
    "Clavicle",
    "Humerus",
    "Radius",
    "Ulna",
    "Meniscus",
    "Labrum",
    "Trochanter",
    "Bursa",
    "Vertebra",

    # Conditions & Diagnoses
    "Osteoarthritis",
    "Rheumatoid Arthritis",
    "Psoriatic Arthritis",
    "Avascular Necrosis",
    "Fracture",
    "Dislocation",
    "Tear",
    "Tendinopathy",
    "Bursitis",
    "Capsulitis",
    "Impingement Syndrome",
    "Calcific Tendonitis",
    "Carpal Tunnel Syndrome",
    "Cubital Tunnel Syndrome",
    "Reynaud's Phenomenon",
    "Sciatica",
    "Radiculopathy",
    "Scoliosis",
    "Lordosis",
    "Kyphosis",
    "Back Pain",
    "Neck Pain",
    "Joint Pain",
    "Hip Pain",
    "Knee Pain",
    "Shoulder Pain",
    "Groin Pain",

    # Procedures & Interventions
    "Orthopedic Procedure",
    "Arthroplasty",
    "Hip Replacement",
    "Knee Replacement",
    "Joint Replacement",
    "Hemiarthroplasty",
    "Arthroscopy",
    "Cheilectomy",
    "Trapeziectomy",
    "Decompression",
    "Osteotomy",
    "Fixation",
    "Injection",
    "Steroid Injection",
    "Plaster Cast",
    "Physiotherapy",
    "Rehabilitation",
    "Surgical Intervention",
    "MRI",
    "X-ray",
    "CT Scan",
    "Ultrasound",

    # UMLS Semantic Types
    "Body Part, Organ, or Organ Component",
    "Tissue",
    "Anatomical Structure",
    "Injury or Poisoning",
    "Disease or Syndrome",
    "Finding",
    "Therapeutic or Preventive Procedure",
    "Diagnostic Procedure"
]

df_sty = df_sty[df_sty['STN'].str.contains('|'.join(ortho_keywords), case=False, na=False)]
print(f"✅ Filtered MRSTY entries: {len(df_sty)}")

# === Merge CONSO + STY ===
df_nodes = df_conso.merge(df_sty[['CUI', 'STN', 'SEMDEF']], on='CUI', how='inner').drop_duplicates('CUI')

# === Load MRDEF: Concept Definitions ===
df_def = pd.read_csv('/YOUR_FILE_LOCATION/MRDEF.RRF.gz', sep='|', header=None, dtype=str)
df_def = df_def.iloc[:, :8]
df_def.columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
df_defs = df_def.groupby('CUI')['DEF'].first().reset_index()

# Merge: Add concept-level definition
df_nodes = df_nodes.merge(df_defs, on='CUI', how='left')
df_nodes = df_nodes[['CUI', 'STR', 'STN', 'SEMDEF', 'DEF']]


# === Generate Embeddings ===
print("🧠 Generating embeddings for concept nodes...")
# Using SapBERT for domain-specific embeddings
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

def embed_node(row):
    text = f"{row['STR']} {row['STN']} {row['DEF']}"
    return model.encode(text, normalize_embeddings=True).tolist()

df_nodes['embedding'] = df_nodes.progress_apply(embed_node, axis=1)

# === Load MRREL ===
df_rel_aa = pd.read_csv('/YOUR_FILE_LOCATION/MRREL.RRF.aa.gz', sep='|', header=None, dtype=str)
df_rel_ac = pd.read_csv('/YOUR_FILE_LOCATION/MRREL.RRF.ac.gz', sep='|', header=None, dtype=str)
df_rel = pd.concat([df_rel_aa, df_rel_ac], ignore_index=True)
df_rel.columns = [
    "CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2",
    "RELA", "RUI", "SRUI", "SAB", "SL", "RG", "DIR", "SUPPRESS", "CVF", "EMPTY"
]

# === Filter only relevant relationships ===
valid_cuis = set(df_nodes['CUI'])
df_edges = df_rel[df_rel['CUI1'].progress_apply(lambda x: x in valid_cuis) &
                  df_rel['CUI2'].progress_apply(lambda x: x in valid_cuis)]
df_edges = df_edges[['CUI1', 'REL', 'CUI2', 'RELA']]

# === Save ===
np.save('umls_orthopaedic_nodes.npy', df_nodes.to_dict('records'))
np.save('umls_orthopaedic_edges.npy', df_edges.to_dict('records'))
print(f"✅ Saved {len(df_nodes)} nodes and {len(df_edges)} edges to .npy files.")
