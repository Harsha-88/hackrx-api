# import os
# import requests
# import tempfile
# import pdfplumber
# from uuid import uuid4
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from tqdm import tqdm

# # ✅ Load environment variables
# load_dotenv()
# HF_API_KEY = os.getenv("HF_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # ✅ Hugging Face Embedding Model (not QA model)
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# def download_pdf(url):
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise Exception("❌ Failed to download PDF.")
    
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#     with open(temp_file.name, 'wb') as f:
#         f.write(response.content)
#     return temp_file.name

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ""
#     return text

# def chunk_text(text, max_tokens=300):
#     paragraphs = text.split('\n')
#     chunks = []
#     current_chunk = ""
#     for para in paragraphs:
#         if len((current_chunk + para).split()) > max_tokens:
#             chunks.append(current_chunk.strip())
#             current_chunk = para
#         else:
#             current_chunk += " " + para
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# def embed_text(texts):
#     headers = {
#         "Authorization": f"Bearer {HF_API_KEY}"
#     }
#     api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

#     embeddings = []
#     for chunk in tqdm(texts, desc="Embedding chunks"):
#         response = requests.post(api_url, headers=headers, json={"inputs": chunk})
#         if response.status_code != 200:
#             raise Exception(f"Embedding failed: {response.text}")
#         vector = response.json()[0]
#         embeddings.append(vector)
#     return embeddings

# def store_in_pinecone(texts, embeddings):
#     pc = Pinecone(api_key=PINECONE_API_KEY)

#     # ✅ Check/create index
#     if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#         print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
#         pc.create_index(
#             name=PINECONE_INDEX_NAME,
#             dimension=len(embeddings[0]),
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )

#     index = pc.Index(PINECONE_INDEX_NAME)

#     vectors = []
#     for i, (text, embed) in enumerate(zip(texts, embeddings)):
#         vectors.append({
#             "id": str(uuid4()),
#             "values": embed,
#             "metadata": {"text": text}
#         })

#     index.upsert(vectors)

# def run_embedding_pipeline(pdf_url):
#     print("📥 Downloading PDF...")
#     pdf_path = download_pdf(pdf_url)

#     print("📄 Extracting text from PDF...")
#     text = extract_text_from_pdf(pdf_path)

#     print("🔗 Splitting into chunks...")
#     chunks = chunk_text(text)

#     print("📊 Embedding chunks...")
#     embeddings = embed_text(chunks)

#     print("📦 Storing in Pinecone...")
#     store_in_pinecone(chunks, embeddings)

#     print("✅ Embedding pipeline complete!")

# # 🧪 For direct testing
# if __name__ == "__main__":
#     sample_pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=..."
#     run_embedding_pipeline(sample_pdf_url)






import os
import requests
import tempfile
import pdfplumber
import numpy as np
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# ✅ Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def download_pdf(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("❌ Failed to download PDF.")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(temp_file.name, 'wb') as f:
        f.write(response.content)
    return temp_file.name

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, max_tokens=300):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len((current_chunk + para).split()) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += " " + para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_text(texts):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

    embeddings = []
    for chunk in tqdm(texts, desc="Embedding chunks"):
        response = requests.post(api_url, headers=headers, json={"inputs": chunk})
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")
        embedding_matrix = response.json()
        vector = np.mean(embedding_matrix, axis=0).tolist()  # ✅ Proper sentence embedding
        embeddings.append(vector)
    return embeddings

def store_in_pinecone(texts, embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []
    for text, embed in zip(texts, embeddings):
        vectors.append({
            "id": str(uuid4()),
            "values": embed,
            "metadata": {"text": text}
        })

    index.upsert(vectors)

def run_embedding_pipeline(pdf_url):
    try:
        print("📥 Downloading PDF...")
        pdf_path = download_pdf(pdf_url)

        print("📄 Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)

        print("🔗 Splitting into chunks...")
        chunks = chunk_text(text)

        print("📊 Embedding chunks...")
        embeddings = embed_text(chunks)

        print("📦 Storing in Pinecone...")
        store_in_pinecone(chunks, embeddings)

        print("✅ Embedding pipeline complete!")
    except Exception as e:
        print(f"❌ Error in embedding pipeline: {str(e)}")

# 🧪 Test locally
if __name__ == "__main__":
    sample_pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=..."
    run_embedding_pipeline(sample_pdf_url)

