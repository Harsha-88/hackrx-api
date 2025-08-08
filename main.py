from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl  # 🛠️ Changed AnyHttpUrl to HttpUrl for OpenAPI compatibility
from typing import List, Dict, Any
from query_handler import run_query
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize app
app = FastAPI(
    title="HackRX Query API",
    description="Query PDFs using Hugging Face QA model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider locking this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

# Health check endpoint
@app.get("/")
def root():
    return {"message": "🚀 HackRX Query API is running."}

# Main POST endpoint
@app.post("/hackrx/run", response_model=Dict[str, Any])
def run_hackrx_query(request: QueryRequest):
    try:
        logging.info("🔵 /hackrx/run API Called!")
        logging.info(f"📄 Input Data: {request.model_dump()}")

        result = run_query(request)
        logging.info("✅ Query Handled Successfully.")
        return result

    except Exception as e:
        logging.exception("❌ Internal Server Error")
        raise HTTPException(status_code=500, detail=str(e))
