import os
from typing import List, Dict
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate
import psycopg2
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

class HybridRAGSystem:
    def __init__(self):
        # Initialize Ollama
        self.llm = Ollama(model="llama 3.2")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Initialize FAISS vector store
        self.vector_dimension = 4096  # Llama 2 embedding dimension
        self.vector_index = faiss.IndexFlatL2(self.vector_dimension)
        
        # Initialize PostgreSQL connection
        self.db_connection_string = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        )
        self.db = SQLDatabase.from_uri(self.db_connection_string)
        
        # Initialize SQL Agent
        self.sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.sql_agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.sql_toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def add_documents_to_vector_store(self, documents: List[str]):
        """Add documents to FAISS vector store"""
        for doc in documents:
            # Get embeddings for the document
            embedding = self.embeddings.embed_query(doc)
            # Add to FAISS index
            self.vector_index.add(np.array([embedding], dtype=np.float32))
    
    def vector_search(self, query: str, k: int = 5) -> List[int]:
        """Search similar documents in vector store"""
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.vector_index.search(query_embedding_array, k)
        return indices[0].tolist()

    def sql_query(self, query: str) -> str:
        """Execute SQL query using the SQL Agent"""
        return self.sql_agent.run(query)

    def hybrid_query(self, query: str) -> Dict:
        """
        Perform hybrid search combining vector and SQL database results
        """
        # Get vector search results
        vector_results = self.vector_search(query)
        
        # Get SQL query results
        sql_results = self.sql_query(query)
        
        # Combine results using LLM
        combination_prompt = PromptTemplate(
            input_variables=["vector_results", "sql_results", "query"],
            template="""
            Given the following search results:
            
            Vector Search Results: {vector_results}
            SQL Database Results: {sql_results}
            
            Please provide a comprehensive answer to the query: {query}
            Combine information from both sources to give the most relevant and complete response.
            """
        )
        
        combination_chain = LLMChain(llm=self.llm, prompt=combination_prompt)
        final_response = combination_chain.run(
            vector_results=vector_results,
            sql_results=sql_results,
            query=query
        )
        
        return {
            "vector_results": vector_results,
            "sql_results": sql_results,
            "combined_response": final_response
        }

if __name__ == "__main__":
    # Initialize the system
    rag_system = HybridRAGSystem()
    
    # Example usage
    documents = [
        "Document 1 content",
        "Document 2 content",
        # Add more documents as needed
    ]
    
    # Add documents to vector store
    rag_system.add_documents_to_vector_store(documents)
    
    # Example hybrid query
    result = rag_system.hybrid_query("Your query here")
    print(result)
