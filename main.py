import os
from dotenv import load_dotenv
from pathlib import Path
from pdf_processor import load_pdf, split_text
from embeddings import create_vector_store, save_vector_store, load_vector_store
from rag import create_rag_chain, query_pdf
import tkinter as tk
from tkinter import filedialog

# Load environment variables from .env file
# This will make API keys accessible via os.environ
load_dotenv()

def select_pdf_file():
    """
    Open a file dialog to select a PDF file and return the path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select PDF File",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    return file_path

def main():
    """
    Main entry point for the PDF RAG application.
    Orchestrates the full workflow:
    1. Load PDF and extract text
    2. Split text into chunks
    3. Generate embeddings for text chunks
    4. Store embeddings in vector database
    5. Process user queries using RAG
    """
    print("PDF RAG Tool - Upload a PDF and ask questions")
    print("=============================================")
    
    # Check if GROQ_API_KEY is set
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found. Please set it in your .env file.")
        return
    
    # Ask if user wants to process a new PDF or use existing vector store
    choice = input("Do you want to: \n1. Process a new PDF \n2. Use existing vector store\nEnter choice (1/2): ")
    
    if choice == "1":
        # Use file dialog to select PDF file
        print("Please select a PDF file in the file dialog...")
        pdf_path = select_pdf_file()
        
        # Check if user canceled the file selection
        if not pdf_path:
            print("File selection canceled. Exiting.")
            return
        
        # Validate if the file exists
        if not Path(pdf_path).exists():
            print(f"Error: File '{pdf_path}' not found.")
            return
        
        print(f"Selected file: {pdf_path}")
        
        # Process PDF and create vector store
        documents = load_pdf(pdf_path)
        chunks = split_text(documents)
        vector_store = create_vector_store(chunks)
        
        # Save vector store
        save_dir = input("Enter a name for the vector store directory (default: vector_store): ") or "vector_store"
        save_vector_store(vector_store, save_dir)
        
    elif choice == "2":
        # Load existing vector store
        save_dir = input("Enter the name of the existing vector store directory (default: vector_store): ") or "vector_store"
        try:
            vector_store = load_vector_store(save_dir)
        except FileNotFoundError:
            print(f"Error: Vector store directory '{save_dir}' not found.")
            return
    else:
        print("Invalid choice. Please run the program again.")
        return
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_store)
    
    # Interactive query loop
    print("\nYou can now ask questions about your PDF. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        # Process query
        result = query_pdf(rag_chain, query)
        
        # Print answer
        print("\nAnswer:")
        print(result["result"])
        
        # Print sources
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Source {i+1}:")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Excerpt: {doc.page_content[:150]}...\n")
    
    print("Thank you for using the PDF RAG Tool!")

if __name__ == "__main__":
    main()

