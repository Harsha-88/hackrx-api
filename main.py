# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# from query_handler import query_document

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class HackRxRequest(BaseModel):
#     documents: str
#     questions: List[str]

# @app.post("/hackrx/run")
# def run_query(payload: HackRxRequest):
#     try:
#         answers = query_document(payload.documents, payload.questions)
#         return {"answers": answers}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from query_handler import run_query
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

# Health check
@app.get("/")
def root():
    return {"message": "🚀 HackRX Query API is running."}

# Main POST endpoint
@app.post("/hackrx/run", response_model=Dict[str, Any])
def run_hackrx_query(request: QueryRequest):
    try:
        logging.info("🔵 /hackrx/run API Called!")
        logging.info(f"📄 Input Data: {request}")

        result = run_query(request)

        logging.info("✅ Query Handled Successfully.")
        return {"status": "success", "data": result}

    except Exception as e:
        logging.error(f"❌ Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

