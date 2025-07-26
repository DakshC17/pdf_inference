import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """Create a vector store from document chunks."""
    print("Creating vector embeddings for document chunks...")
    
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model initialized: sentence-transformers/all-MiniLM-L6-v2")
    
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created with {len(chunks)} embedded chunks")
    
    return vector_store

## loading file

def save_vector_store(vector_store, directory="vector_store"):
    """Save vector store to disk."""
    os.makedirs(directory, exist_ok=True)
    vector_store.save_local(directory)
    print(f"Vector store saved to '{directory}' directory")

def load_vector_store(directory="vector_store"):
    """Load vector store from disk."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Vector store directory '{directory}' not found")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
   
    vector_store = FAISS.load_local(directory, embeddings)
    print(f"Vector store loaded from '{directory}' directory")
    
    return vector_store




if __name__ == "__main__":
  
    from pdf_processor import load_pdf, split_text
    
    test_pdf_path = input("Enter a PDF path to test embeddings: ")
    
    
    docs = load_pdf(test_pdf_path)
    chunks = split_text(docs)
    
    vs = create_vector_store(chunks)
    save_dir = "test_vector_store"
    save_vector_store(vs, save_dir)
    
    loaded_vs = load_vector_store(save_dir)
    
    query = input("Enter a test query to search in the document: ")
    results = loaded_vs.similarity_search(query, k=2)
    
    print("\nSearch Results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")
    
    print("\nEmbeddings module test complete!")
