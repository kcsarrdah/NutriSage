from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from datetime import datetime
import time

# Define the PDF files with descriptions
pdf_files = {
    'a': {
        'path': "data/Dietary_Guidelines_for_Americans_2020-2025.pdf",
        'description': "Official Dietary Guidelines",
        'use_plumber': False
    },
    'b': {
        'path': "data/Healthy_Dinners_Cookbook.pdf",
        'description': "Healthy Dinner Recipes",
        'use_plumber': False
    },
    'c': {
        'path': "data/High_Protein_Recipe_Pack_One.pdf",
        'description': "High Protein Recipes Part 1",
        'use_plumber': False
    },
    'd': {
        'path': "data/High_Protein_Recipe_Pack_Two.pdf",
        'description': "High Protein Recipes Part 2",
        'use_plumber': False
    },
    'e': {
        'path': "data/Nutritive_Value_of_Foods.pdf",
        'description': "Nutritional Values Database",
        'use_plumber': True
    }
}

def create_vector_db():
    start_time = time.time()
    print(f"Starting vector database creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all PDFs
    documents = []
    total_pages = 0
    
    for source_id, file_info in pdf_files.items():
        try:
            if os.path.exists(file_info['path']):
                print(f"\nProcessing {file_info['description']} ({source_id})...")
                
                # Choose loader based on file configuration
                loader = PDFPlumberLoader(file_info['path']) if file_info['use_plumber'] else PyPDFLoader(file_path=file_info['path'])
                docs = loader.load()
                
                # Enhance metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_id': source_id,
                        'description': file_info['description'],
                        'filename': os.path.basename(file_info['path']),
                        'processing_date': datetime.now().isoformat()
                    })
                
                documents.extend(docs)
                total_pages += len(docs)
                print(f"Successfully loaded {len(docs)} pages")
                
            else:
                print(f"Warning: File {file_info['path']} not found")
        
        except Exception as e:
            print(f"Error processing {file_info['path']}: {str(e)}")
            continue

    print(f"\nTotal loaded: {total_pages} pages from {len(documents)} documents")

    # Split documents into sections
    print("\nSplitting documents into sections...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    sections = text_splitter.split_documents(documents)
    print(f"Created {len(sections)} sections")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    print("Generating embeddings...")
    section_texts = [section.page_content for section in sections]
    embeddings = model.encode(section_texts, show_progress_bar=True)
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Create and save FAISS index
    print("\nCreating FAISS index...")
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    # Ensure database directory exists
    os.makedirs("database", exist_ok=True)

    # Save the index
    print("Saving FAISS index...")
    faiss.write_index(index, "database/pdf_sections_index.faiss")

    # Prepare and save sections data
    print("Saving sections data...")
    sections_data = [
        {
            'content': section.page_content,
            'metadata': section.metadata
        }
        for section in sections
    ]

    with open('database/pdf_sections_data.pkl', 'wb') as f:
        pickle.dump(sections_data, f)

    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print summary
    print("\nVector Database Creation Summary:")
    print("-" * 50)
    print(f"Total documents processed: {len(pdf_files)}")
    print(f"Total pages processed: {total_pages}")
    print(f"Total sections created: {len(sections)}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Database files saved to: {os.path.abspath('database')}")
    print("-" * 50)

if __name__ == "__main__":
    create_vector_db()
