from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import tempfile
import os
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import logging

# Load env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Logging
logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI(
    title="PDF Query API",
    description="Upload PDFs and ask questions with HuggingFace models + Pinecone",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔐 Lock to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class QueryInput(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        os.remove(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")

        # Basic chunking
        chunks = text.split("\n\n")
        embeddings = model.encode(chunks).tolist()

        ids = [str(uuid4()) for _ in chunks]
        vectors = list(zip(ids, embeddings, [{"text": c} for c in chunks]))

        index.upsert(vectors=vectors)

        return {
            "message": "✅ PDF uploaded and embedded successfully.",
            "chunks": len(chunks)
        }

    except Exception as e:
        logging.exception("Error while uploading PDF")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def ask_question(query: QueryInput):
    try:
        question_embedding = model.encode(query.question).tolist()

        search_result = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        answers = [match['metadata']['text'] for match in search_result['matches']]

        return {
            "question": query.question,
            "answers": answers
        }

    except Exception as e:
        logging.exception("Error while querying")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "🚀 API is live and running!"}
