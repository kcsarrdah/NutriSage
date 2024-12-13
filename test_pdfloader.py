from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
import os

def test_pdf_loader():
    # Define the PDF files
    pdf_files = {
        'a': "data/Dietary_Guidelines_for_Americans_2020-2025.pdf",
        'b': "data/Healthy_Dinners_Cookbook.pdf",
        'c': "data/High_Protein_Recipe_Pack_One.pdf",
        'd': "data/High_Protein_Recipe_Pack_Two.pdf",
        'e': "data/Nutritive_Value_of_Foods.pdf"
    }
    
    total_pages = 0
    results = {}
    
    print("\nTesting PDF Loading...")
    print("-" * 50)
    
    # Test each PDF file
    for source_id, pdf_path in pdf_files.items():
        try:
            if os.path.exists(pdf_path):
                # Try PDFPlumberLoader for the problematic file
                if source_id == 'e':
                    loader = PDFPlumberLoader(pdf_path)
                else:
                    loader = PyPDFLoader(file_path=pdf_path)
                
                docs = loader.load()
                
                # Store results
                results[source_id] = {
                    'status': 'Success',
                    'pages': len(docs),
                    'metadata_sample': docs[0].metadata if docs else None
                }
                total_pages += len(docs)
                
                print(f"File {source_id}:")
                print(f"  - Path: {pdf_path}")
                print(f"  - Pages: {len(docs)}")
                print(f"  - First page metadata: {docs[0].metadata}")
                print(f"  - Content preview: {docs[0].page_content[:100]}...")
                print("-" * 50)
            else:
                results[source_id] = {
                    'status': 'File not found',
                    'pages': 0,
                    'metadata_sample': None
                }
                print(f"File {source_id}:")
                print(f"  - Path: {pdf_path}")
                print(f"  - Status: File not found")
                print("-" * 50)
                
        except Exception as e:
            results[source_id] = {
                'status': f'Error: {str(e)}',
                'pages': 0,
                'metadata_sample': None
            }
            print(f"File {source_id}:")
            print(f"  - Path: {pdf_path}")
            print(f"  - Status: Error - {str(e)}")
            print("-" * 50)
    
    # Print summary
    print("\nSummary:")
    print(f"Total PDFs attempted: {len(pdf_files)}")
    print(f"Successfully loaded PDFs: {sum(1 for r in results.values() if r['status'] == 'Success')}")
    print(f"Total pages loaded: {total_pages}")
    
    return results

if __name__ == "__main__":
    test_pdf_loader()