# app.py (Streamlit Frontend)
import streamlit as st
import pandas as pd
import requests
import io
import time

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# ---------- Apply Orange-Grey Theme ----------
st.markdown("""
<style>
    /* Your existing CSS theme */
    :root {
        --primary: #FF8C00;
        --secondary: #FFA500;
        --accent: #FFB74D;
        --dark: #2C3E50;
        --medium: #34495E;
        --light: #ECF0F1;
    }
    .stApp {
        background: linear-gradient(135deg, #ECF0F1 0%, #FFFFFF 100%);
    }
    .stButton > button {
        background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- API Client Functions ----------
def call_fast_api(file_content: bytes, filename: str):
    files = {"file": (filename, file_content, "text/csv")}
    response = requests.post(f"{API_BASE_URL}/run_fast", files=files)
    return response.json()

def call_config1_api(file_content: bytes, filename: str, config: dict):
    files = {"file": (filename, file_content, "text/csv")}
    data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
    response = requests.post(f"{API_BASE_URL}/run_config1", files=files, data=data)
    return response.json()

def call_deep_api(file_content: bytes, filename: str, config: dict):
    files = {"file": (filename, file_content, "text/csv")}
    data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
    response = requests.post(f"{API_BASE_URL}/run_deep", files=files, data=data)
    return response.json()

def call_retrieve_api(query: str, k: int = 5):
    data = {"query": query, "k": k}
    response = requests.post(f"{API_BASE_URL}/retrieve", data=data)
    return response.json()

# ---------- Streamlit App ----------
st.set_page_config(page_title="Chunking Optimizer", layout="wide", page_icon="📦")

st.markdown("""
<div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">📦 Chunking Optimizer</h1>
    <p style="color: white; text-align: center; margin: 10px 0 0 0; font-size: 1.2em;">With Retrieval Testing</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "api_results" not in st.session_state:
    st.session_state.api_results = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = None

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0;">API Status</h2>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.success("✅ API Connected")
    except:
        st.error("❌ API Not Connected")

    if st.session_state.api_results:
        st.markdown("### 📊 Last Results")
        result = st.session_state.api_results
        st.write(f"**Mode:** {result.get('mode', 'N/A')}")
        if 'summary' in result:
            st.write(f"**Chunks:** {result['summary'].get('chunks', 'N/A')}")
            st.write(f"**Storage:** {result['summary'].get('stored', 'N/A')}")
            if result['summary'].get('retrieval_ready'):
                st.success("🔍 Retrieval Ready")

    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Mode selection
st.markdown("## 🎯 Choose Processing Mode")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚡ Fast Mode", use_container_width=True):
        st.session_state.current_mode = "fast"
with col2:
    if st.button("⚙️ Config-1 Mode", use_container_width=True):
        st.session_state.current_mode = "config1"
with col3:
    if st.button("🔬 Deep Config Mode", use_container_width=True):
        st.session_state.current_mode = "deep"

if st.session_state.current_mode:
    st.success(f"Selected: **{st.session_state.current_mode.upper()} MODE**")

# File upload
st.markdown("### 📤 Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ **{uploaded_file.name}** loaded!")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.markdown(f"""
        <div style="background: #ECF0F1; padding: 15px; border-radius: 8px;">
            <p><strong>Dataset Info:</strong></p>
            <p>• Rows: {len(df)}</p>
            <p>• Columns: {len(df.columns)}</p>
        </div>
        """, unsafe_allow_html=True)

# Mode-specific processing
if st.session_state.current_mode and st.session_state.uploaded_file:
    if st.session_state.current_mode == "fast":
        st.markdown("### ⚡ Fast Mode")
        if st.button("🚀 Run Fast Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running Fast Mode..."):
                result = call_fast_api(
                    st.session_state.uploaded_file.getvalue(),
                    st.session_state.uploaded_file.name
                )
                st.session_state.api_results = result
                st.success("✅ Fast pipeline completed!")

    elif st.session_state.current_mode == "config1":
        st.markdown("### ⚙️ Config-1 Mode")
        col1, col2 = st.columns(2)
        with col1:
            null_handling = st.selectbox("Null handling", ["keep", "drop", "fill"])
            fill_value = st.text_input("Fill value", "Unknown") if null_handling == "fill" else None
            chunk_method = st.selectbox("Chunk method", ["fixed", "recursive", "semantic", "document"])
        with col2:
            model_choice = st.selectbox("Model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"])
            storage_choice = st.selectbox("Storage", ["faiss", "chromadb"])
        
        if st.button("🚀 Run Config-1 Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running Config-1..."):
                config = {
                    "null_handling": null_handling,
                    "fill_value": fill_value,
                    "chunk_method": chunk_method,
                    "chunk_size": 400,
                    "overlap": 50,
                    "model_choice": model_choice,
                    "storage_choice": storage_choice
                }
                result = call_config1_api(
                    st.session_state.uploaded_file.getvalue(),
                    st.session_state.uploaded_file.name,
                    config
                )
                st.session_state.api_results = result
                st.success("✅ Config-1 pipeline completed!")

    elif st.session_state.current_mode == "deep":
        st.markdown("### 🔬 Deep Config Mode")
        col1, col2 = st.columns(2)
        with col1:
            null_handling = st.selectbox("Null handling", ["keep", "drop", "fill"], key="deep_null")
            fill_value = st.text_input("Fill value", "Unknown", key="deep_fill") if null_handling == "fill" else None
            remove_stopwords = st.checkbox("Remove stopwords")
            lowercase = st.checkbox("Lowercase", value=True)
        with col2:
            stemming = st.checkbox("Stemming")
            lemmatization = st.checkbox("Lemmatization")
            chunk_method = st.selectbox("Chunk method", ["fixed", "recursive", "semantic", "document"], key="deep_chunk")
            model_choice = st.selectbox("Model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"], key="deep_model")
            storage_choice = st.selectbox("Storage", ["faiss", "chromadb"], key="deep_storage")
        
        if st.button("🚀 Run Deep Config Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running Deep Config..."):
                config = {
                    "null_handling": null_handling,
                    "fill_value": fill_value,
                    "remove_stopwords": remove_stopwords,
                    "lowercase": lowercase,
                    "stemming": stemming,
                    "lemmatization": lemmatization,
                    "chunk_method": chunk_method,
                    "chunk_size": 400,
                    "overlap": 50,
                    "model_choice": model_choice,
                    "storage_choice": storage_choice
                }
                result = call_deep_api(
                    st.session_state.uploaded_file.getvalue(),
                    st.session_state.uploaded_file.name,
                    config
                )
                st.session_state.api_results = result
                st.success("✅ Deep Config pipeline completed!")

# Retrieval Section
if st.session_state.api_results and st.session_state.api_results.get('summary', {}).get('retrieval_ready'):
    st.markdown("---")
    st.markdown("## 🔍 Test Retrieval")
    st.markdown("Test how well your chunks work with semantic search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your search query:", placeholder="Search for similar content...")
    with col2:
        k = st.slider("Top K results", 1, 10, 3)
    
    if query:
        with st.spinner("Searching..."):
            retrieval_result = call_retrieve_api(query, k)
            st.session_state.retrieval_results = retrieval_result
            
            if "error" in retrieval_result:
                st.error(f"Retrieval error: {retrieval_result['error']}")
            else:
                st.success(f"✅ Found {len(retrieval_result['results'])} results")
                
                for result in retrieval_result['results']:
                    similarity_color = "#28a745" if result['similarity'] > 0.7 else "#ffc107" if result['similarity'] > 0.4 else "#dc3545"
                    
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {similarity_color};">
                        <h4 style="margin: 0 0 10px 0; color: {similarity_color};">
                            Rank #{result['rank']} (Similarity: {result['similarity']:.3f})
                        </h4>
                        <p style="margin: 0; color: #666; font-size: 0.9em;">{result['content'][:300]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>📦 Chunking Optimizer • FastAPI + Streamlit • With Retrieval Testing</p>
</div>
""", unsafe_allow_html=True)