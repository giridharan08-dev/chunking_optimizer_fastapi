# main.py
from fastapi import FastAPI, UploadFile, File, Form
from backend import run_fast_pipeline, run_config1_pipeline, run_deep_pipeline, retrieve_similar
import pandas as pd
import io

app = FastAPI(title="Chunking Optimizer API", version="1.0")

# ---------------------------
# FAST MODE
# ---------------------------
@app.post("/run_fast")
async def run_fast(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = run_fast_pipeline(df)
    return {"mode": "fast", "summary": result}

# ---------------------------
# CONFIG-1 MODE
# ---------------------------
@app.post("/run_config1")
async def run_config1(
    file: UploadFile = File(...),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("all-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss")
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = run_config1_pipeline(df, null_handling, fill_value, chunk_method,
                                  chunk_size, overlap, model_choice, storage_choice)
    return {"mode": "config1", "summary": result}

# ---------------------------
# DEEP CONFIG MODE
# ---------------------------
@app.post("/run_deep")
async def run_deep(
    file: UploadFile = File(...),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    remove_stopwords: bool = Form(False),
    lowercase: bool = Form(True),
    stemming: bool = Form(False),
    lemmatization: bool = Form(False),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("all-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss")
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = run_deep_pipeline(df, null_handling, fill_value, remove_stopwords,
                               lowercase, stemming, lemmatization, chunk_method,
                               chunk_size, overlap, model_choice, storage_choice)
    return {"mode": "deep", "summary": result}

# ---------------------------
# RETRIEVAL ENDPOINT
# ---------------------------
@app.post("/retrieve")
async def retrieve(query: str = Form(...), k: int = Form(5)):
    """Retrieve similar chunks after running any pipeline"""
    result = retrieve_similar(query, k)
    return result

# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Chunking Optimizer API is running", "version": "1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}