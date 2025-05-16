import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """Create a vector store from document chunks."""
    print("Creating vector embeddings for document chunks...")
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model initialized: sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created with {len(chunks)} embedded chunks")
    
    return vector_store