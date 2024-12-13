from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import faiss
import numpy as np
import pickle

# Load the FAISS index
try:
    index = faiss.read_index("database/pdf_sections_index.faiss")
except FileNotFoundError:
    print("FAISS index file not found. Please ensure 'pdf_sections_index.faiss' exists.")
    exit(1)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load sections data
try:
    with open('database/pdf_sections_data.pkl', 'rb') as f:
        sections_data = pickle.load(f)
except FileNotFoundError:
    print("Sections data file not found. Please ensure 'pdf_sections_data.pkl' exists.")
    exit(1)

def search_faiss(query, k=3):
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'distance': dist,
            'content': sections_data[idx]['content'],
            'metadata': sections_data[idx]['metadata']
        })
    
    return results

# Create a prompt template

prompt_template = """
You are a nutrition expert with access to a food database. For questions about specific foods:
1. First state the exact nutritional values found in the data
2. Distinguish between different variations (e.g., single vs double patty)
3. Include portion sizes when available
4. List all nutrients mentioned in the data

Context:
{context}

Question: {question}

Provide nutritional information in a clear, structured format:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = Ollama(
    model="llama3"
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

def answer_question(query):
    # Search for relevant context
    search_results = search_faiss(query)
    
    # Combine the content from the search results
    context = "\n\n".join([result['content'] for result in search_results])

    # Run the chain
    response = chain.run(context=context, question=query)
    
    return response

# Example usage
query = "What are the main dietary guidelines for protein intake?"
answer = answer_question(query)

print(f"Question: {query}")
print(f"Answer: {answer}")
