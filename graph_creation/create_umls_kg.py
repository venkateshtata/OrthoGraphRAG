from py2neo import Graph, Node
import numpy as np
import pandas as pd
from tqdm import tqdm

# === Parameters ===
SRDEF_PATH = '/YOUR_FILE_LOCATION/SRDEF'

# === Load Data ===
nodes = np.load('../YOUR_FILE_LOCATION/umls_orthopaedic_nodes.npy', allow_pickle=True).tolist()
edges = np.load('../YOUR_FILE_LOCATION/umls_orthopaedic_edges.npy', allow_pickle=True).tolist()

valid_cuis = set(n['CUI'] for n in nodes)
edges = [e for e in edges if e["CUI1"] in valid_cuis and e["CUI2"] in valid_cuis]

# === Load SRDEF for readable relationship mapping ===
df_srdef = pd.read_csv(SRDEF_PATH, sep='|', header=None, dtype=str)
df_srdef.columns = ["Type", "Code", "Name", "TreeNumber", "Definition", "5", "6", "7", "Abbrev", "9", "10"]

rel_def_map = df_srdef[df_srdef["Type"] == "REL"].set_index("Code")["Definition"].to_dict()
rel_label_map = {
    "RB": "is broader than",
    "RN": "is narrower than",
    "CHD": "is a child of",
    "PAR": "is a parent of",
    "SY": "is synonymous with"
}

# === Connect to Neo4j ===
graph = Graph("bolt://localhost:7690", auth=("neo4j", "password"))

# WARNING: This deletes everything!
graph.run("MATCH (n) DETACH DELETE n")
print('GRAPH CLEARED!')

# Create constraint for UMLS Concepts (public graph)
graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")

# === Build CUI → name mapping ===
cui_to_name = {n["CUI"]: n["STR"] for n in nodes}

# === Insert UMLS Concept Nodes ===
print(f"🧠 Inserting {len(nodes)} UMLS concept nodes...")
skipped = 0
for node_data in tqdm(nodes):
    name = node_data.get("STR")
    if pd.isna(name) or name.strip().lower() == "nan" or not name.strip():
        skipped += 1
        continue

    definition=node_data.get("DEF")
    if pd.isna(definition) or definition.strip().lower() == "nan" or not definition.strip():
        skipped += 1
        continue


    node = Node(
        "Concept",
        namespace = "medical_ns",
        name=name,
        CUI=node_data.get("CUI"),
        semantic_type=node_data.get("STN"),
        semantic_definition=node_data.get("SEMDEF"),
        definition=node_data.get("DEF"),
        embedding=node_data.get("embedding")
    )

    graph.merge(node, "Concept", "name")

print(f"✅ Skipped {skipped} nodes with missing names.")

# === Insert Relationships with readable labels ===
print(f"🔗 Inserting {len(edges)} UMLS relationships...")
for edge_data in tqdm(edges):
    if edge_data.get("CUI1") == edge_data.get("CUI2"):
        continue

    name1 = cui_to_name.get(edge_data.get("CUI1"))
    name2 = cui_to_name.get(edge_data.get("CUI2"))
    if not name1 or not name2:
        continue

    rel = edge_data.get("REL")
    rela = edge_data.get("RELA")

    rel = rel.strip().upper().replace(" ", "_") if isinstance(rel, str) else None
    rela = rela.strip() if isinstance(rela, str) else None

    if not rel:
        continue

    raw_label = rel_label_map.get(rel)
    if raw_label is None:
        continue

    relationship_type = raw_label.upper().replace(" ", "_")
    description = rel_def_map.get(rel)

    params = {
        "name1": name1,
        "name2": name2,
        "rel": rel,
        "label": raw_label
    }
    prop_lines = ["rel: $rel", "label: $label"]

    if rela:
        params["rela"] = rela
        prop_lines.append("rela: $rela")
    if description:
        params["description"] = description
        prop_lines.append("description: $description")

    query = f"""
    MATCH (a:Concept {{name: $name1}}), (b:Concept {{name: $name2}})
    MERGE (a)-[r:{relationship_type} {{ {', '.join(prop_lines)} }}]->(b)
    """

    graph.run(query, parameters=params)

print(f"\n✅ UMLS Graph complete — with labels logically separated via `:Concept`.")
