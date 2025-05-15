import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# This will make API keys accessible via os.environ
load_dotenv()

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
    
    # Get user input for PDF file path
    pdf_path = input("Enter the path to your PDF file: ")
    
    # Validate if the file exists
    if not Path(pdf_path).exists():
        print(f"Error: File '{pdf_path}' not found.")
        return