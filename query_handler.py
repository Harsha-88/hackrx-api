import os
import tempfile
import requests
import pdfplumber
import re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

if not HF_API_KEY:
    raise EnvironmentError("❌ Hugging Face API key not found in .env file.")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# Sentence embedding model
embedder = SentenceTransformer("thenlper/gte-small")

# ✅ Extract and clean sentences from PDF
def extract_sentences_from_pdf(pdf_url: str):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"❌ Failed to download PDF: {response.status_code}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    os.remove(tmp_path)

    # Split into clean sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    return clean_sentences[:300]  # Limit for performance

# ✅ Find best context sentence for QA
def get_best_context(question, sentences):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
    best_index = int(similarities.argmax())
    return sentences[best_index]

# ✅ Hugging Face QA call
def ask_question(context: str, question: str) -> str:
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "").strip()
    except Exception as e:
        return f"❌ Error during inference: {str(e)}"

# ✅ Main query handler
def run_query(request):
    try:
        sentences = extract_sentences_from_pdf(request.documents)
        
        if not sentences:
            return {
                "status": "error",
                "message": "No sentences extracted from the PDF."
            }

        answers = []
        for question in request.questions:
            context = get_best_context(question, sentences)
            answer = ask_question(context, question)

            if answer and answer.lower() not in ["", "no answer found"]:
                answers.append(answer)
            else:
                answers.append("No answer found.")

        return {
            "status": "success",
            "data": {
                "answers": answers
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
