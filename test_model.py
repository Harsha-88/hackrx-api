import os
import requests
from dotenv import load_dotenv

# ✅ Load .env file
load_dotenv()

# ✅ Get token
API_TOKEN = os.getenv("HF_API_KEY")

# ✅ Check token value before sending request
if not API_TOKEN:
    print("❌ Token not found. Check .env file.")
    exit()

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

data = {
    "inputs": {
        "question": "What is the capital of India?",
        "context": "India is a country in South Asia. Its capital is New Delhi."
    }
}

response = requests.post(API_URL, headers=headers, json=data)

print(f"🟡 Status Code: {response.status_code}")
try:
    print(f"🟢 Response JSON: {response.json()}")
except Exception as e:
    print(f"🔴 JSON Error: {e}")
    print(f"🔴 Response Text: {response.text}")
