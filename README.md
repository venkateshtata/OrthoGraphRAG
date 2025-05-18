# OrthoGraphRAG

OrthoGraphRAG is a specialized medical information retrieval system that combines knowledge graph capabilities with Retrieval-Augmented Generation (RAG) for orthopedic medical data processing. It leverages both structured knowledge graphs and unstructured document retrieval to provide comprehensive and accurate responses to medical queries.

## Features

- **Hybrid Retrieval System**: Combines knowledge graph traversal and vector similarity search
- **Multiple Data Sources**:
  - **UMLS Medical Knowledge Graph**: Utilizes UMLS (Unified Medical Language System) for standardized medical terminology
  - **Private Knowledge Graph**: Stores patient-specific clinical data and relationships
  - **Vector Database**: Implements semantic search on medical documents
- **Configurable Data Source Selection**: Toggle between different data sources as needed
- **Advanced Entity Extraction**: Uses LLMs to identify medical entities from unstructured text
- **Medical-Domain Specific Embeddings**: Employs SapBERT medical embeddings for semantic search 
- **Comprehensive Knowledge Integration**: Connects private entities to UMLS concepts through semantic similarity

## Project Structure

- **`graph_creation/`**: Tools for building the knowledge graphs
  - `create_umls_kg.py`: Creates a Neo4j graph from UMLS medical concepts
  - `get_umls_nodes_edges.py`: Extracts and processes relevant orthopedic nodes/relationships from UMLS
  - `create_private_kg.py`: Processes clinical documents to extract entities and relationships
  - `connect_entities.py`: Links private entities to UMLS concepts using semantic similarity
- **`retriever.py`**: Main interface for retrieving information and generating responses

## Setup and Installation

### Prerequisites

- Python 3.8+
- Neo4j graph database
- PostgreSQL with pgvector extension (for RAG functionality)
- Ollama (for local LLM capabilities)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/OrthoGraphRAG.git
   cd OrthoGraphRAG
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure Neo4j:
   - Start a Neo4j instance (default port is 7687)
   - Set username/password (defaults in the code are "neo4j"/"strongpass123")

4. Configure PostgreSQL with pgvector for document storage:
   - Follow pgvector installation instructions
   - Create a database with vector extension enabled

5. Set up Ollama:
   - Install Ollama and download preferred model (default is llama2:70b)

## Usage

### Creating Knowledge Graphs

1. **UMLS Graph Creation**:
   ```
   cd graph_creation
   python get_umls_nodes_edges.py  # Extract orthopedic-relevant nodes
   python create_umls_kg.py  # Load nodes into Neo4j
   ```

2. **Private Knowledge Graph**:
   ```
   python create_private_kg.py  # Process clinical documents
   python connect_entities.py  # Link to UMLS concepts
   ```

### Running the Retriever

```python
from retriever import hybrid_retriever

# Configure data sources in retriever.py as needed:
# USE_PRIVATE_GRAPH_DATA = True/False
# USE_RAG_DATA = True/False
# USE_UMLS_GRAPH_DATA = True/False

response = hybrid_retriever("What treatments has Mrs. Holmes received for her osteoarthritis?")
print(response)
```

## Configuration

The main configuration parameters are in `retriever.py`:

- **Data Source Selection**:
  - `USE_PRIVATE_GRAPH_DATA`: Include patient-specific graph data
  - `USE_RAG_DATA`: Include document chunks from vector search
  - `USE_UMLS_GRAPH_DATA`: Include UMLS medical concepts

- **Neo4j Connection**:
  - `NEO4J_URI`: Connection URI (default: "bolt://localhost:7687")
  - `NEO4J_USER`: Username
  - `NEO4J_PASSWORD`: Password

- **Ollama Configuration**:
  - `OLLAMA_MODEL`: Model to use (default: "llama2:70b")
  - `OLLAMA_URL`: API endpoint (default: "http://localhost:11434/api/generate")

- **Retrieval Parameters**:
  - Various constants for limiting result counts


## Acknowledgments

- UMLS (Unified Medical Language System) for medical terminology data
- SapBERT for medical domain-specific embeddings
