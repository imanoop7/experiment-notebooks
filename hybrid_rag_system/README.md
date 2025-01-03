# Hybrid RAG System

This is a Hybrid Retrieval-Augmented Generation (RAG) system that combines vector search using FAISS and structured data queries using PostgreSQL, powered by Langchain and Ollama (Llama 2).

## Features

- Vector-based similarity search using FAISS
- Structured data querying using PostgreSQL
- Langchain SQL Agent for natural language queries
- Hybrid search combining both vector and SQL results
- Powered by Ollama's Llama 2 model

## Prerequisites

1. Python 3.8+
2. PostgreSQL database
3. Ollama installed and running locally

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and update the values:
```bash
cp .env.example .env
```

5. Initialize the database:
```bash
python init_db.py
```

## Usage

1. Start Ollama with Llama 2 model:
```bash
ollama run llama2
```

2. Run the hybrid RAG system:
```python
from hybrid_rag import HybridRAGSystem

# Initialize the system
rag_system = HybridRAGSystem()

# Add documents to vector store
documents = ["Your document content here"]
rag_system.add_documents_to_vector_store(documents)

# Perform hybrid query
result = rag_system.hybrid_query("Your query here")
print(result)
```

## System Architecture

The system combines two types of retrieval:
1. **Vector Search**: Documents are embedded using Llama 2 and stored in FAISS for similarity search
2. **SQL Database**: Structured data is stored in PostgreSQL and queried using Langchain's SQL Agent

The results from both sources are combined using an LLM to provide comprehensive answers to queries.

## License

MIT
