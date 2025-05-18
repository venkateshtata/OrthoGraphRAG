import ast
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from py2neo import Graph

# === CONFIGURATION ===
NEO4J_URI = "bolt://localhost:7690"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
SIMILARITY_THRESHOLD = 0.5  # Adjust based on your requirements

# === CONNECT TO GRAPH ===
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === FETCH NODES WITH EMBEDDINGS ===
print("🔍 Fetching embedded private entities...")
private_nodes = graph.run("""
    MATCH (e:Entity)
    WHERE e.embedding IS NOT NULL
    RETURN e.id AS id, e.name AS name, e.embedding AS embedding
""").data()

print("📚 Fetching UMLS concept nodes...")
umls_nodes = graph.run("""
    MATCH (c:Concept)
    WHERE c.embedding IS NOT NULL
    RETURN c.name AS name, c.CUI AS cui, c.embedding AS embedding
""").data()

if not private_nodes or not umls_nodes:
    print("❌ No embedded nodes found. Please ensure embeddings are stored.")
    exit()

# === EMBEDDING PARSING HELPER ===
def parse_embedding(embedding):
    # If the embedding is a string, parse it; otherwise, assume it's already a list.
    if isinstance(embedding, str):
        return np.array(ast.literal_eval(embedding))
    return np.array(embedding)

# --- Option 1: Filter Out Inconsistent Embeddings ---
expected_dim = 768  # Change this if you want the other dimension

# Filter private nodes to only include those with the expected dimension.
private_nodes_filtered = []
private_embeddings_list = []
for node in private_nodes:
    emb = parse_embedding(node['embedding'])
    if len(emb) == expected_dim:
        private_nodes_filtered.append(node)
        private_embeddings_list.append(emb)
    else:
        print(f"Skipping private node {node['id']} with embedding dimension {len(emb)}")

# Optionally, if you need to filter UMLS nodes similarly, do so. Here we assume they are consistent.
umls_embeddings_list = [parse_embedding(n['embedding']) for n in umls_nodes]

private_embeddings = np.array(private_embeddings_list)
umls_embeddings = np.array(umls_embeddings_list)

# --- Option 2: Alternatively, pad/truncate embeddings to a uniform dimension ---
# Uncomment the block below if you prefer padding/truncating rather than filtering.
"""
def pad_embedding(emb, target_dim):
    emb = np.array(emb)
    if emb.shape[0] < target_dim:
        return np.pad(emb, (0, target_dim - emb.shape[0]), 'constant')
    elif emb.shape[0] > target_dim:
        return emb[:target_dim]
    return emb

target_dim = 768  # Use 768 as the uniform dimension
private_embeddings = np.array([pad_embedding(parse_embedding(n['embedding']), target_dim) for n in private_nodes])
umls_embeddings = np.array([pad_embedding(parse_embedding(n['embedding']), target_dim) for n in umls_nodes])
private_nodes_filtered = private_nodes  # In this option, we are not filtering nodes out.
"""

# === PERFORM COSINE SIMILARITY CALCULATION ===
print("🔗 Calculating similarities...")
similarities = cosine_similarity(private_embeddings, umls_embeddings)

# === CREATE SEMANTIC LINKS ABOVE THRESHOLD ===
print("🧠 Creating semantic relationships...")
for i, private_node in enumerate(tqdm(private_nodes_filtered)):
    for j, umls_node in enumerate(umls_nodes):
        sim_score = similarities[i, j]
        if sim_score >= SIMILARITY_THRESHOLD:
            graph.run("""
                MATCH (a:Entity {id: $private_id})
                MATCH (b:Concept {CUI: $umls_cui})
                MERGE (a)-[r:REFERENCE_OF]->(b)
                SET r.similarity = $score
            """, parameters={
                "private_id": private_node["id"],
                "umls_cui": umls_node["cui"],
                "score": float(sim_score)
            })

print("✅ Semantic similarity connections established.")
