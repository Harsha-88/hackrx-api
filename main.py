from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import tempfile
import os
import requests
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize FastAPI app
app = FastAPI()

# Allow Swagger UI and other frontend tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class QueryInput(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    os.remove(tmp_path)

    chunks = text.split("\n\n")  # Simple chunking
    embeddings = model.encode(chunks).tolist()

    ids = [str(uuid4()) for _ in chunks]
    to_upsert = list(zip(ids, embeddings, [{"text": c} for c in chunks]))
    index.upsert(vectors=to_upsert)

    return {"message": "PDF uploaded and embedded successfully.", "chunks": len(chunks)}


@app.post("/query")
def ask_question(query: QueryInput):
    question = query.question
    question_embedding = model.encode(question).tolist()

    search_result = index.query(vector=question_embedding, top_k=5, include_metadata=True)
    answers = [match['metadata']['text'] for match in search_result['matches']]

    return {
        "question": question,
        "answers": answers
    }


@app.get("/")
def home():
    return {"message": "API is running!"}
