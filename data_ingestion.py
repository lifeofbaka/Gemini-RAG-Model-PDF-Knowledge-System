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


"""
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt_tab')"""

# === get all text files 

pattern = "./Data/*/text/*"
files = glob.glob(pattern, recursive=True)

files = [file for file in files if ".json" not in file]  # remove json files from initial search 


def hash_table(table_html: str) -> str:
    """

    This function creates a hash from the table html contents.

    :param table_html: table html contents
    :type table_html: str
    :return: hash of the table html contents
    :rtype: str
    
    """
    return hashlib.sha256(table_html.encode("utf-8")).hexdigest()


def document_chunking(file: str) -> None: 
    """
    This function creates chunks from markdown files and breaks up data 

    :param file: file path 
    :type file: str
    
    """

    # === Load markdown file ===
    loader = UnstructuredMarkdownLoader(
        file,
        mode="elements",
        strategy="fast",
    )

    doc = loader.load()

    # === locate and store document points by category === 
    raw_tables = {} 
    text = {}
    for item in range(len(doc)): 
        category = doc[item].metadata.get("category") 

        # Seperate Tables from other text categories, titles, and lists 
        if 'Table' in category:
            raw_tables[item] = doc[item]
        else:
            text[item] = doc[item]

    # === check if there are any tables 
    if len(raw_tables) > 0:     

        # === check if there is document table summary data is already stored === 

        table_out = file.split('/')[-1].split('.')[0] + "_table_summaries.json" # table file name
        cur_dir = pathlib.Path(file).parent
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
                                            "element_id": table_doc.metadata.get("element_id"),
                                            "source": table_doc.metadata.get("source"),
                                            "model": "gemini-2.0-flash-001",
                                            }
            

        with open(table_out_path, "w") as f:
            json.dump(table_summaries, f, indent=2)
    






        




"""    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=512,        # Set your desired chunk size
    chunk_overlap=50)      # Set your desired overlap (10% is a good start)
    """


document_chunking(files[0])

