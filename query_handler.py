

# import os
# import tempfile
# import requests
# import pdfplumber
# import numpy as np
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer

# # ✅ Load environment variables
# load_dotenv()
# HF_API_KEY = os.getenv("HF_API_KEY")
# HF_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# # ✅ Validate token
# if not HF_API_KEY:
#     raise EnvironmentError("❌ Hugging Face API key not found in .env file.")

# HEADERS = {
#     "Authorization": f"Bearer {HF_API_KEY}",
#     "Content-Type": "application/json"
# }

# # ✅ SentenceTransformer for chunk similarity
# embedder = SentenceTransformer("thenlper/gte-small")

# # ✅ Extract and split text from PDF
# def extract_text_from_pdf(pdf_url: str):
#     response = requests.get(pdf_url)
#     if response.status_code != 200:
#         raise Exception(f"❌ Failed to download PDF: {response.status_code}")
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(response.content)
#         tmp_path = tmp.name

#     chunks = []
#     with pdfplumber.open(tmp_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 # Only include long-enough line
#                  lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
#                  chunks.extend(lines)

#     os.remove(tmp_path)
    
#     if not chunks:
#         raise ValueError("❌ No valid text extracted from the PDF.")
    
#     return chunks[:100]  # ✅ Optional: limit to 100 chunks for speed

# # ✅ Find most relevant context chunk using dot product similarity
# def get_relevant_chunk(question, chunks):
#     question_embedding = embedder.encode([question])[0]
#     chunk_embeddings = embedder.encode(chunks)
#     similarities = np.dot(chunk_embeddings, question_embedding)
#     top_idx = int(np.argmax(similarities))
#     return chunks[top_idx]

# # ✅ Ask the model (deepset/roberta-base-squad2)
# def ask_question(context: str, question: str) -> str:
#     payload = {
#         "inputs": {
#             "question": question,
#             "context": context
#         }
#     }

#     try:
#         response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
#         response.raise_for_status()
#         result = response.json()

#         return result.get("answer", "No answer found.").strip()
#     except Exception as e:
#         return f"❌ Error during inference: {str(e)}"

# # ✅ Main query runner
# def run_query(request):
#     try:
#         chunks = extract_text_from_pdf(request.documents)
#         answers = []

#         for question in request.questions:
#             context = get_relevant_chunk(question, chunks)
#             answer = ask_question(context, question)
#             answers.append(answer)

#         return {"answers": answers}

#     except Exception as e:
#         return {"error": str(e)}































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
