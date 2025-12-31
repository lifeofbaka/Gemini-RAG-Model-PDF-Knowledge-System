# === LLM and Embeddings ===
# import langchain 
# import transformers
# import chromadb

# === Google GenAI ===
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma

import chromadb
from dotenv import load_dotenv

# === Frontend ===
# import gradio as gr 

# === System Utilities ===
import os 
import sys
import pathlib

# === Load Secrets ===
from dotenv import load_dotenv
load_dotenv() 

api_key = os.environ["GOOGLE_GENAI_API_KEY"]

# === Initialize Google GenAI Client ===
client = genai.Client(api_key=api_key)

# Example: Generate content using Gemini 2.0 Flash model
"""response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents={'text': 'Why is the sky blue?'},
    config={
        'stop_sequences': ['\n'], # stop generation at newline
        'max_output_tokens': 50, # maximum length of the output
        'temperature': 0, # creativity level
        'top_p': 0.95, # nucleus sampling
        'top_k': 20, # top-k sampling
    },
)

print(response.text) """


load_dotenv()

api_key = os.environ["GOOGLE_GENAI_API_KEY"]

model = model = init_chat_model("google_genai:gemini-2.5-flash-lite", api_key=api_key)

# === embeddings === 

# TODO ==== Update to google doc method of loading embeddings === 
# to build using google doc a class object must be built 
# the langchain object below is a pydantic class object which contains client initialization and 
# embedding creation methods ""

# initialized client 
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=api_key)

# === Setup Vectore store and Persistent Storage  ===

if not os.path.exists("./chroma_db"):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_client.create_collection(
    name="example_collection",
    embedding_function=embeddings,
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200
        }
    }
)
else: 
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
