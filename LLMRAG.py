# === LLM and Embeddings ===
# import langchain 
# import transformers
# import chromadb

# === Google GenAI ===
from google import genai

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

