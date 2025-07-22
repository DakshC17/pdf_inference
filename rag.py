import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_llm():
    """Set up the Groq LLM for generation."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"  
    )
    
    print(f"LLM initialized with model: llama3-8b-8192")
    return llm

def create_rag_chain(vector_store):
    """Create a RAG chain with the vector store and LLM."""
    llm = setup_llm()
    
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  
    )
    
    
    template = """You are an AI assistant providing information based on the given documents.
    
    Answer the user's question based ONLY on the following context:
    {context}
    
    Question: {question}
    
    If the answer cannot be determined from the context, say "I don't have enough information to answer that question based on the provided documents."
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("RAG chain created successfully")
    return rag_chain

def query_pdf(rag_chain, query):
    """Query the PDF using the RAG chain."""
    print(f"Querying with: '{query}'")
    
    result = rag_chain({"query": query})
    
    
    return result

if __name__ == "__main__":
    
    from embeddings import load_vector_store
    
    
    try:
        vector_store = load_vector_store("test_vector_store")
        
        
        chain = create_rag_chain(vector_store)
        
        
        while True:
            query = input("\nEnter a question (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            
            result = query_pdf(chain, query)
            
            
            print("\nAnswer:")
            print(result["result"])
            
            
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"]):
                print(f"Source {i+1}:")
                print(f"Page: {doc.metadata.get('page', 'Unknown')}")
                print(f"Excerpt: {doc.page_content[:150]}...\n")
        
    except FileNotFoundError:
        print("No vector store found. Please run embeddings.py first to create a vector store.")
