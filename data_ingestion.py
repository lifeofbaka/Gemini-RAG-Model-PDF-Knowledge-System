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
import re

from tqdm import tqdm

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

    # file directory
    cur_dir = pathlib.Path(file).parent

    # Load md file 
    loader = UnstructuredMarkdownLoader(
        file,
        mode="elements",
        strategy="fast",
    )

    doc = loader.load()


    #  === File output paths === 
    text_out_path = os.path.join(
        cur_dir, pathlib.Path(file).stem + "_text_chunks.json"
    )
    table_out_path = os.path.join(
        cur_dir, pathlib.Path(file).stem + "_table_summaries.json"
    )

    # === Load caches ===
    text_summaries = {}
    if os.path.exists(text_out_path):
        with open(text_out_path) as f:
            text_summaries = json.load(f)

    table_summaries = {}
    if os.path.exists(table_out_path):
        with open(table_out_path) as f:
            table_summaries = json.load(f)

    # =========================
    # PAGE TRACKING STATE
    # =========================
    current_page = None
    page_pattern = re.compile(r"^page:\s*(\d+)", re.IGNORECASE)

    # =========================
    # PROCESS ELEMENTS IN ORDER
    # =========================
    for element in tqdm(doc, desc="Processing elements"):
        text = element.page_content.strip()
        if not text:
            continue

        # Detect page marker
        page_match = page_pattern.search(text)
        if page_match:
            current_page = int(page_match.group(1))
            continue  # Do NOT store page markers as content

        category = element.metadata.get("category", "UncategorizedText")

        # =========================
        # TABLE HANDLING
        # =========================
        if "Table" in category:
            table_html = element.metadata.get("text_as_html")
            table_hash = hash_table(table_html)

            if table_hash in table_summaries:
                continue
            
            # === create summary of markdown table ===
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents={"text": table_html},
                config={
                    "max_output_tokens": 1000,
                    "temperature": 0,
                    "system_instruction": (
                        "You are an expert assistant that summarizes tables accurately, "
                        "preserving all key information."
                    ),
                },
            )


            table_summaries[table_hash] = {
                "page_content": response.text, 
                
                "metadata": {"table_html": table_html,
                             "source": element.metadata.get("source"),
                             "element_id": element.metadata.get("element_id"),
                             "category": category,
                             "document_id": element.metadata.get("filename"),
                             "parent_id": element.metadata.get("parent_id"),
                             "page": current_page,
                             "language": element.metadata.get("languages"),
                             "model": "gemini-2.0-flash-001",}
            }
            # print(table_summaries)

        # =========================
        # TEXT HANDLING
        # =========================
        else:
            text_hash = hash_text(text)
            if text_hash in text_summaries:
                continue

            text_summaries[text_hash] = {
                "page_content": text,
                  "metadata" : {  "source": element.metadata.get("source"),
                                "element_id": element.metadata.get("element_id"),
                                "category": category,
                                "document_id": element.metadata.get("filename"),
                                "page": current_page,
                                "language": element.metadata.get("languages"),
                                "parent_id": element.metadata.get("parent_id"),
                }
                
            }

            # print(text_summaries)

    # =========================
    # WRITE OUTPUTS
    # =========================
    with open(text_out_path, "w") as f:
        json.dump(text_summaries, f, indent=2)

    with open(table_out_path, "w") as f:
        json.dump(table_summaries, f, indent=2)

# === Usage ===
for file in files: 
    cache_text_and_table_artifacts_from_markdown(file)

