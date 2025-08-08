import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import tempfile

# HuggingFace API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# FastAPI app setup
app = FastAPI(title="HackRX Query API", version="1.0")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class UploadRequest(BaseModel):
    documents: str  # PDF URL

class QueryRequest(BaseModel):
    query: str

# Memory (Simple in-memory DB)
pdf_text = ""

# Route to upload and embed PDF
@app.post("/hackrx/run")
def upload_pdf(req: UploadRequest):
    global pdf_text

    pdf_url = req.documents
    try:
        # Download PDF from URL
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid PDF URL.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Extract text using pdfplumber
        with pdfplumber.open(tmp_file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        pdf_text = text
        return {"message": "PDF uploaded and text extracted successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Route to query uploaded PDF
@app.post("/hackrx/query")
def query_pdf(req: QueryRequest):
    global pdf_text

    if not pdf_text:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")

    query = req.query

    try:
        # Call Hugging Face API for embedding
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        hf_payload = {
            "inputs": [query, pdf_text],
        }

        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=hf_payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Embedding API failed.")

        embeddings = response.json()
        query_embedding = embeddings[0]
        doc_embedding = embeddings[1]

        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm

        sim = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
        match = "Yes" if sim > 0.75 else "No"

        return {
            "matched": match,
            "similarity_score": round(sim, 2),
            "extracted_text": pdf_text[:300] + "...",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/")
def read_root():
    return {"status": "API is live!"}
