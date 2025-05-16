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



def save_vector_store(vector_store, directory="vector_store"):
    """Save vector store to disk."""
    os.makedirs(directory, exist_ok=True)
    vector_store.save_local(directory)
    print(f"Vector store saved to '{directory}' directory")

def load_vector_store(directory="vector_store"):
    """Load vector store from disk."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Vector store directory '{directory}' not found")
    
    # Initialize the same embedding model used for creation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store
    vector_store = FAISS.load_local(directory, embeddings)
    print(f"Vector store loaded from '{directory}' directory")
    
    return vector_store