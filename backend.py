# backend.py
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import chromadb
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Global variables to store current state for retrieval
current_model = None
current_store_info = None
current_chunks = None
current_embeddings = None

# -----------------------------
# 🔹 Preprocessing
# -----------------------------
def preprocess_basic(df: pd.DataFrame, null_handling="keep", fill_value=None):
    if null_handling == "drop":
        df = df.dropna().reset_index(drop=True)
    elif null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    return df

def preprocess_advanced(df: pd.DataFrame,
                        null_handling="keep", fill_value=None,
                        remove_stopwords=False, lowercase=False,
                        stemming=False, lemmatization=False):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    stop_words = set(stopwords.words("english")) if remove_stopwords else set()
    ps = PorterStemmer() if stemming else None
    lm = WordNetLemmatizer() if lemmatization else None

    df = preprocess_basic(df, null_handling, fill_value)

    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].astype(str)

        if lowercase:
            series = series.str.lower()

        if remove_stopwords or stemming or lemmatization:
            new_vals = []
            for text in series:
                tokens = re.findall(r"\w+", text)
                if remove_stopwords:
                    tokens = [t for t in tokens if t not in stop_words]
                if stemming:
                    tokens = [ps.stem(t) for t in tokens]
                if lemmatization:
                    tokens = [lm.lemmatize(t) for t in tokens]
                new_vals.append(" ".join(tokens))
            series = pd.Series(new_vals)

        df[col] = series

    return df

# -----------------------------
# 🔹 Chunking
# -----------------------------
def chunk_fixed(df: pd.DataFrame, chunk_size=400, overlap=50):
    text = "\n".join(df.astype(str).agg(" | ".join, axis=1).tolist())
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def chunk_recursive_keyvalue(df: pd.DataFrame, chunk_size=400, overlap=50):
    rows = []
    for _, row in df.iterrows():
        kv_pairs = [f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c])]
        rows.append(" | ".join(kv_pairs))
    big_text = "\n".join(rows)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(big_text)

def chunk_document_based(df: pd.DataFrame):
    return df.astype(str).agg(" | ".join, axis=1).tolist()

def chunk_semantic_cluster(df: pd.DataFrame, n_clusters=5):
    """Group rows into clusters based on semantic embeddings of rows."""
    sentences = df.astype(str).agg(" ".join, axis=1).tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(sentences)
    kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42)
    labels = kmeans.fit_predict(embs)

    grouped = {}
    for sent, lab in zip(sentences, labels):
        grouped.setdefault(lab, []).append(sent)

    return [" ".join(v) for v in grouped.values()]

# -----------------------------
# 🔹 Embedding + Storage
# -----------------------------
def embed_texts(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return model, np.array(embeddings).astype("float32")

def store_chroma(chunks, embeddings, collection_name="chunks_collection"):
    client = chromadb.PersistentClient(path="chromadb_store")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )
    return {"type": "chroma", "collection": col, "collection_name": collection_name}

def store_faiss(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return {"type": "faiss", "index": index}

# -----------------------------
# 🔹 Retrieval Functions
# -----------------------------
def retrieve_similar(query: str, k: int = 5):
    """Retrieve similar chunks using the current stored embeddings"""
    global current_model, current_store_info, current_chunks, current_embeddings
    
    if current_model is None or current_store_info is None:
        return {"error": "No model or store available. Run a pipeline first."}
    
    # Encode query
    query_embedding = current_model.encode([query])
    query_arr = np.array(query_embedding).astype("float32")
    
    results = []
    
    if current_store_info["type"] == "faiss":
        # FAISS retrieval
        index = current_store_info["index"]
        distances, indices = index.search(query_arr, k)
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(current_chunks):
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                results.append({
                    "rank": i + 1,
                    "content": current_chunks[idx],
                    "similarity": float(similarity),
                    "distance": float(distance)
                })
    
    elif current_store_info["type"] == "chroma":
        # Chroma retrieval
        collection = current_store_info["collection"]
        chroma_results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "distances"]
        )
        
        for i, (doc, distance) in enumerate(zip(
            chroma_results["documents"][0], 
            chroma_results["distances"][0]
        )):
            similarity = 1 / (1 + distance)
            results.append({
                "rank": i + 1,
                "content": doc,
                "similarity": float(similarity),
                "distance": float(distance)
            })
    
    else:
        # Fallback: cosine similarity with raw embeddings
        if current_embeddings is not None:
            similarities = cosine_similarity(query_arr, current_embeddings)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            for i, idx in enumerate(top_indices):
                results.append({
                    "rank": i + 1,
                    "content": current_chunks[idx],
                    "similarity": float(similarities[idx]),
                    "distance": float(1 - similarities[idx])
                })
    
    return {"query": query, "k": k, "results": results}

# -----------------------------
# 🔹 Pipelines
# -----------------------------
def run_fast_pipeline(df):
    global current_model, current_store_info, current_chunks, current_embeddings
    
    # Auto preprocess
    df1 = preprocess_basic(df, null_handling="drop")
    # Default: semantic clustering
    chunks = chunk_semantic_cluster(df1)
    model, embs = embed_texts(chunks)
    store = store_faiss(embs)
    
    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    return {
        "rows": len(df1), 
        "cols": len(df1.columns), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "retrieval_ready": True
    }

def run_config1_pipeline(df, null_handling, fill_value, chunk_method,
                         chunk_size, overlap, model_choice, storage_choice):
    global current_model, current_store_info, current_chunks, current_embeddings
    
    df1 = preprocess_basic(df, null_handling, fill_value)

    if chunk_method == "fixed":
        chunks = chunk_fixed(df1, chunk_size, overlap)
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue(df1, chunk_size, overlap)
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster(df1)
    else:
        chunks = chunk_document_based(df1)

    model, embs = embed_texts(chunks, model_choice)
    
    if storage_choice == "faiss":
        store = store_faiss(embs)
    else:
        store = store_chroma(chunks, embs, f"config1_{int(pd.Timestamp.now().timestamp())}")

    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "retrieval_ready": True
    }

def run_deep_pipeline(df, null_handling, fill_value, remove_stopwords,
                      lowercase, stemming, lemmatization,
                      chunk_method, chunk_size, overlap, model_choice, storage_choice):
    global current_model, current_store_info, current_chunks, current_embeddings
    
    df1 = preprocess_advanced(df, null_handling, fill_value,
                              remove_stopwords, lowercase, stemming, lemmatization)

    if chunk_method == "fixed":
        chunks = chunk_fixed(df1, chunk_size, overlap)
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue(df1, chunk_size, overlap)
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster(df1)
    else:
        chunks = chunk_document_based(df1)

    model, embs = embed_texts(chunks, model_choice)
    
    if storage_choice == "faiss":
        store = store_faiss(embs)
    else:
        store = store_chroma(chunks, embs, f"deep_{int(pd.Timestamp.now().timestamp())}")

    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "retrieval_ready": True
    }