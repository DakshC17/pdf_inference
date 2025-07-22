from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    
    print(f"Loading PDF from: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Successfully loaded {len(documents)} pages from the PDF.")
    
    if documents:
        print("\nSample text from first page:")
        print(f"{documents[0].page_content[:150]}...")
        print(f"Metadata: {documents[0].metadata}")
    
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better processing."""
    print(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} text chunks.")
    
    if chunks:
        print("\nSample chunk:")
        print(f"Content: {chunks[0].page_content[:100]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    return chunks

if __name__ == "__main__":
    test_pdf_path = input("Enter a PDF path to test the processor: ")
    docs = load_pdf(test_pdf_path)
    chunks = split_text(docs)
    print(f"\nProcessing complete. Generated {len(chunks)} chunks from {len(docs)} pages.")
