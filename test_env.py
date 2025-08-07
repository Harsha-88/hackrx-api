# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

print("PINECONE_API_KEY =", os.getenv("PINECONE_API_KEY"))
print("PINECONE_ENVIRONMENT =", os.getenv("PINECONE_ENVIRONMENT"))
print("PINECONE_INDEX_NAME =", os.getenv("PINECONE_INDEX_NAME"))
print("OPENROUTER_API_KEY =", os.getenv("OPENROUTER_API_KEY"))
