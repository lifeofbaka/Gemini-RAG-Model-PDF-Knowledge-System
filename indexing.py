import json 
import glob 

from LLMRAG import chroma_client
from uuid import uuid4

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

import os 
import getpass
from pydantic import BaseModel, SecretStr



# === Load JSON Artifacts === 
# TODO === Uodate loading of JSON Files === 
pattern = "./Data/*/text/*.json"
files = glob.glob(pattern, recursive=True)

print(f"Found {len(files)} files.")
if not files:
    print(f"No files found matching {pattern}. Check your current directory: {os.getcwd()}")




# Load documents 


documents = []
for file in files: 
    try:
        loader = JSONLoader(
            file_path=file,
            jq_schema= ".[] | .page_content", # ".messages[].content",
            text_content=True,
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from {file}")
        documents.extend(docs)
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Total documents loaded: {len(documents)}")
if not documents:
    print("No documents loaded. Check if your jq_schema matches the JSON structure.")


# text splits 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # chunk size (characters)
    chunk_overlap=100,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = [] 
all_splits = text_splitter.split_documents(documents)

print(f"Total splits generated: {len(all_splits)}")


# === Add to Vector Store ===
print(all_splits[0])

chroma_client.get_collection("example_collection").add(
    documents=[doc.page_content for doc in all_splits],
    metadatas=[doc.metadata for doc in all_splits],
    ids=[str(uuid4()) for _ in all_splits])

print("Documents added to ChromaDB collection 'example_collection'.")