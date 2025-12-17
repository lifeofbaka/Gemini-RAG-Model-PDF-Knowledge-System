import langchain 
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from LLMRAG import client

# === System Utils === 
import os 
import pathlib
import glob

import json
import hashlib

# === get all text files 

pattern = "./Data/*/text/*"
files = glob.glob(pattern, recursive=True)

files = [file for file in files if ".json" not in file]  # remove json files from initial search 

def hash_text(content: str) -> str:
    """
    
    This function creates a hash from the text contents.

    :param content: text contents
    :type content: str
    :return: hash of the text contents
    :rtype: str
    
    """
    return hashlib.sha256(
        content.strip().encode("utf-8")
    ).hexdigest()

def hash_table(table_html: str) -> str:
    """

    This function creates a hash from the table html contents.

    :param table_html: table html contents
    :type table_html: str
    :return: hash of the table html contents
    :rtype: str
    
    """
    return hashlib.sha256(table_html.encode("utf-8")).hexdigest()


def cache_text_and_table_artifacts_from_markdown(file: str) -> None: 
    """
    This function parses text and table data from markdown file and creats usable 
    json artifacts.

    :param file: file path 
    :type file: str
    
    """
    cur_dir = pathlib.Path(file).parent

    # === Load markdown file ===
    loader = UnstructuredMarkdownLoader(
        file,
        mode="elements",
        strategy="fast",
    )

    doc = loader.load()

    # === locate and store document points by category === 
    raw_tables = {} 
    raw_text = {}
    for item in range(len(doc)): 
        category = doc[item].metadata.get("category") 

        # Seperate Tables from other text categories, titles, and lists 
        if 'Table' in category:
            raw_tables[item] = doc[item]
        else:
            raw_text[item] = doc[item]

    text_out = file.split('/')[-1].split('.')[0] + "_text_chunks.json" # text file name
    text_out_path = os.path.join(cur_dir, text_out)


    # =========================
    # TEXT EXTRACTION (NO LLM)
    # =========================
    text_summaries = {}
    if os.path.exists(text_out_path):
        with open(text_out_path, "r") as f:
            text_summaries = json.load(f)
    else:
        text_summaries = {}

    for key, text_doc in raw_text.items():

        text_content = text_doc.page_content.strip()
        if not text_content:
            continue

        text_hash = hash_text(text_content)

        if text_hash in text_summaries:
                print("Text was already extracted...")
                continue
        
        # text data
        parent_id = text_doc.metadata.get("parent_id") 
        document_id = text_doc.metadata.get("filename")

        text_summaries[text_hash] = {"text": text_content,
                                    "source": text_doc.metadata.get("source"),
                                    "element_id": text_doc.metadata.get("element_id"),
                                    "category": text_doc.metadata.get("category"),
                                    "document_id": document_id,
                                    "page": text_doc.metadata.get("page"),
                                    "language": text_doc.metadata.get("languages"),
                                        }
        
        if parent_id:
            text_summaries["parent_id"] = parent_id
        

    with open(text_out_path, "w") as f:
        json.dump(text_summaries, f, indent=2)



    # =========================
    # TABLE SUMMARIZATION (LLM)
    # =========================
    if not raw_tables:
        return    

    # === check if there is document table summary data is already stored === 

    table_out = file.split('/')[-1].split('.')[0] + "_table_summaries.json" # table file name
    
    table_out_path = os.path.join(cur_dir, table_out)

        # Load Tables or Create new tables 
    if os.path.exists(table_out_path):
        with open(table_out_path, "r") as f:
            table_summaries = json.load(f)
    else:
        table_summaries = {}

    for key, table_doc in raw_tables.items():

        table_html = table_doc.metadata.get("text_as_html")
        table_hash = hash_table(table_html)

        # Skip if already summarized
        if table_hash in table_summaries:
            print("Table was already summarized, skipping...")
            continue
        
        # Create table summary
        response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents={"text": table_html},
                    config={
                        "max_output_tokens": 1000,
                        "temperature": 0,
                        "top_p": 0.95,
                        "top_k": 20, 
                        "system_instruction": (
                            "You are an expert assistant that summarizes tables accurately, "
                            "preserving all key information."),
                        },
                    )
        # store table summary 
        table_summaries[table_hash] = { "summary": response.text,
                                        "source": table_doc.metadata.get("source"),
                                        "element_id": table_doc.metadata.get("element_id"),
                                        "parent_id": table_doc.metadata.get("parent_id"), 
                                        "category": table_doc.metadata.get("category"),
                                        "document_id": table_doc.metadata.get("filename"),
                                        "page": table_doc.metadata.get("page"),
                                        "language": table_doc.metadata.get("languages"),
                                        "model": "gemini-2.0-flash-001",
                                        }
        

    with open(table_out_path, "w") as f:
        json.dump(table_summaries, f, indent=2)
    
# === Usage ===
for file in files: 
    cache_text_and_table_artifacts_from_markdown(file)

