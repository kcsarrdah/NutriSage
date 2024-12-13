import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import faiss
import numpy as np
import pickle

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Custom theme configurations
def get_theme_config():
    themes = {
        'light': {
            'bg_color': '#ffffff',
            'secondary_bg_color': '#f0f2f6',
            'text_color': '#0e1117',
            'primary_color': '#4CAF50',
            'secondary_color': '#45a049'
        },
        'dark': {
            'bg_color': '#0e1117',
            'secondary_bg_color': '#1e2128',
            'text_color': '#ffffff',
            'primary_color': '#2ea043',
            'secondary_color': '#238636'
        }
    }
    return themes[st.session_state.theme]

# Set page config
st.set_page_config(page_title="NutriSage", page_icon="🍎", layout="wide")

# Get current theme colors
theme = get_theme_config()

# Apply theme CSS
st.markdown(f"""
    <style>
        /* Main containers */
        section[data-testid="stSidebar"] > div {{
            background-color: {theme['secondary_bg_color']};
        }}
        
        section[data-testid="stSidebar"] .block-container {{
            background-color: {theme['secondary_bg_color']};
        }}
        
        .stApp {{
            background-color: {theme['bg_color']};
        }}
        
        header[data-testid="stHeader"] {{
            background-color: {theme['bg_color']};
        }}
        
        .main .block-container {{
            background-color: {theme['bg_color']};
        }}
        
        .big-font {{
            font-size: 20px !important;
            font-weight: bold;
            color: {theme['text_color']};
        }}
        
        .stButton>button {{
            background-color: {theme['primary_color']};
            color: white;
            font-size: 18px;
            width: 100%;
            border-radius: 6px;
        }}
        
        .stExpander {{
            border: 1px solid {theme['primary_color']};
            border-radius: 5px;
        }}
        
        .app-description {{
            color: {theme['text_color']} !important;
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 20px;
            padding: 1rem;
            background-color: {theme['secondary_bg_color']};
            border-radius: 8px;
        }}
        
        .footer-text {{
            color: {theme['text_color']} !important;
            text-align: center;
            font-size: 14px;
            margin-top: 20px;
        }}
        
        /* Text Elements */
        h1, h2, h3, h4, h5, h6, .stMarkdown {{
            color: {theme['text_color']} !important;
        }}
        
        .css-10trblm, .css-16idsys p {{
            color: {theme['text_color']} !important;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            background-color: {theme['secondary_bg_color']};
            color: {theme['text_color']};
        }}

        /* Sidebar specific styles */
        [data-testid="stSidebarNav"] svg {{
            fill: {theme['text_color']};
        }}
        
        .sidebar-content {{
            background-color: {theme['secondary_bg_color']};
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}

        .sidebar-section {{
            margin-bottom: 1.5rem;
        }}

        .sidebar-title {{
            color: {theme['text_color']};
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}

        .sidebar-text {{
            color: {theme['text_color']};
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        /* Fix sidebar arrow color */
        button[data-testid="baseButton-header"] svg {{
            fill: {theme['text_color']};
        }}
        
    </style>
""", unsafe_allow_html=True)

# Load the FAISS index
@st.cache_resource
def load_faiss_index():
    try:
        return faiss.read_index("database/pdf_sections_index.faiss")
    except FileNotFoundError:
        st.error("FAISS index file not found. Please ensure 'pdf_sections_index.faiss' exists.")
        st.stop()

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load sections data
@st.cache_data
def load_sections_data():
    try:
        with open('database/pdf_sections_data.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Sections data file not found. Please ensure 'pdf_sections_data.pkl' exists.")
        st.stop()

# Initialize resources
index = load_faiss_index()
model = load_embedding_model()
sections_data = load_sections_data()

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

def new_search_faiss(query, k=3, threshold=1):
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:
            results.append({
                'distance': dist,
                'content': sections_data[idx]['content'],
                'metadata': sections_data[idx]['metadata']
            })
    
    return results

prompt_template = """
You are an AI assistant specialized in dietary guidelines. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

@st.cache_resource
def load_llm():
    return Ollama(model="llama3")

llm = load_llm()
chain = LLMChain(llm=llm, prompt=prompt)

def answer_question(query):
    search_results = new_search_faiss(query)
    if search_results==[]:
        response="I don't know. The question is most likely out of context."
        context = ""
    else:
        context = "\n\n".join([result['content'] for result in search_results])
        response = chain.run(context=context, question=query)
    return response, context

# Enhanced sidebar with instructions and app information
with st.sidebar:
    # Theme Settings
    st.markdown("### 🎨 Theme Settings")
    if st.button("🌓 Toggle Theme" if st.session_state.theme == 'light' else "☀️ Toggle Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

    # About Section
    st.markdown("### 📱 About NutriSage")
    st.markdown("Your AI-powered nutrition assistant that provides evidence-based dietary guidance.")

    # What You Can Do Section
    st.markdown("### 🎯 What You Can Do")
    st.markdown("""
    • Ask questions about nutrition
    • Learn about food groups
    • Get dietary recommendations
    • Understand portion sizes
    • Explore healthy eating patterns
    """)

    # How to Use Section
    st.markdown("### 💡 How to Use")
    st.markdown("""
    1. Type your nutrition question
    2. Click "Get Answer" button
    3. View the detailed context if needed
    4. Adjust theme for better viewing
    """)

    # Data Source Section
    st.markdown("### 📚 Data Source")
    st.markdown("Based on the Dietary Guidelines for Americans 2020-2025.")

# Main UI
st.title("🍽️ NutriSage: Dietary Guidelines Q&A")

st.markdown('<p class="big-font">Ask a question about dietary guidelines:</p>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g., What are the main food groups?")

if st.button("Get Answer"):
    if query:
        with st.spinner("Searching and generating answer..."):
            answer, context = answer_question(query)
            st.subheader("Answer:")
            st.info(answer)
            with st.expander("Show Context"):
                st.write(context)
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown("Source: [Dietary Guidelines for Americans 2020-2025](https://www.dietaryguidelines.gov/sites/default/files/2020-12/Dietary_Guidelines_for_Americans_2020-2025.pdf)")

# Simplified footer
st.markdown("---")
st.markdown("""
    <div class="footer-text">
        Powered by AI and built with Ollama, Streamlit, and LangChain<br>
        Your trusted companion for making informed dietary choices<br>
        Stay healthy, eat smart! 🥗
    </div>
    """, unsafe_allow_html=True)