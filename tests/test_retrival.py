# test_retrieval.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load resources
index = faiss.read_index("database/pdf_sections_index.faiss")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('database/pdf_sections_data.pkl', 'rb') as f:
    sections_data = pickle.load(f)

def test_search(query, k=5):
    # Encode query
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.expand_dims(query_vector, axis=0)
    
    # Search
    distances, indices = index.search(query_vector, k)
    
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    # Show results
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        result = sections_data[idx]
        print(f"\nResult {i+1}:")
        print(f"Distance: {dist:.4f}")
        print(f"Source: {result['metadata'].get('description', 'Unknown')}")
        print(f"Page: {result['metadata'].get('page', 'Unknown')}")
        print("\nContent Preview:")
        print(result['content'][:300], "...\n")
        print("-" * 50)

if __name__ == "__main__":
    # Test queries
    queries = [
        "Calories in a cheeseburger",
        "Is a cheeseburger good for me?",
        "nutritional value of burger",
        "caloric content fast food"
    ]
    
    for query in queries:
        test_search(query)