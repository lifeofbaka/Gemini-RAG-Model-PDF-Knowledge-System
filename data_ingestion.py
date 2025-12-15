import langchain 
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader



# === System Utils === 
import os 
import pathlib
import glob


import nltk

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


    
"""    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=512,        # Set your desired chunk size
    chunk_overlap=50)      # Set your desired overlap (10% is a good start)
    """


document_chunking(files[0])

