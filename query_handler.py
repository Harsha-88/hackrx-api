



import os
import tempfile
import requests
import pdfplumber
import numpy as np
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ✅ Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

if not HF_API_KEY:
    raise EnvironmentError("❌ Hugging Face API key not found in .env file.")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# ✅ Sentence embedding model for semantic similarity
embedder = SentenceTransformer("thenlper/gte-small")


# ✅ Extract and clean sentences from a PDF
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

    # Split into clean, informative sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    return clean_sentences[:300]  # Limit for performance


# ✅ Get the most relevant sentence as context
def get_best_context(question, sentences):
    question_embedding = embedder.encode([question])[0]
    sentence_embeddings = embedder.encode(sentences)
    similarities = np.dot(sentence_embeddings, question_embedding)
    best_index = int(np.argmax(similarities))
    return sentences[best_index]


# ✅ Call Hugging Face QA model
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


# ✅ Main function to handle a list of questions
def run_query(request):
    try:
        sentences = extract_sentences_from_pdf(request.documents)
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
