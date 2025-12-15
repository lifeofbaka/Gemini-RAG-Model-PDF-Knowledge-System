# === PDF and Document Processing ===
import pymupdf.layout
import pymupdf4llm


# === System Utilities ===
import os 
import sys
import pathlib
from tqdm import tqdm


# === PDF Files ===
file_paths = []
data = pathlib.Path("./Google Reports")
for file in data.iterdir():
    file_paths.append(pathlib.Path(file)) # file path


# === PDF Extraction Function ===
def pdf_extract(file_paths: list, text_extraction=True, image_extraction=True) -> None:
    """
    * This function Extracts contents of a PDF document(s). 
    
        * It will extract text and table data into a markdown file. 
        * Images will be stored as jpg. 
    
    
    :param file_paths: list of file paths to PDF documents
    :type file_paths: list
    """

    if type(file_paths) != list: 
        raise Exception("File paths must be a list.")
    
    # === Loop through files === 
    for file in tqdm(file_paths, desc="Processing Files"):

        file_name = pathlib.Path(file).name.split(".")[0] # get file name without extension"

        suffix = ".md" # or ".json" or ".txt" 

        text_out = pathlib.Path(f"./Data/{file_name}/text/{file_name}_out").with_suffix(suffix) # set file output 
        image_out = pathlib.Path(f"./Data/{file_name}/images")

        # === Check if file text already processed ===
        if text_extraction:
            if pathlib.Path(text_out).exists():
                print(f"{file_name} text already processed.")
                print("Skipping...")
            else:
                doc = pymupdf.open(file) # Load document 
                try:
                    os.makedirs(f"./Data/{file_name}/text")
                    print(f"Directory for {file_name} text files created.")
                except FileExistsError:
                    print(f"Directory for {file_name} text files already exists.")
                
                # === Text and Table Extraction ===
                print (f"Extracting text and table data from {file_name}...")

                md = pymupdf4llm.to_markdown(doc)

                text_out.write_bytes(md.encode()) # write markdown file to data directory
                print(f"Text and Tables extracted from {file_name} and saved to {text_out.with_suffix(suffix)}")
        else: 
            doc = None
            

        if image_extraction:

            # === Helper Function ===
            def image_helper():
                nonlocal doc
                if doc == None: 
                    doc = pymupdf.open(file) # Load document 
                try:
                    os.makedirs(f"./Data/{file_name}/images")
                    print(f"Directory for {file_name} image files created.")
                except FileExistsError:
                    print(f"Directory for {file_name} image files already exists.")
                

                # === Image Extraction ===
                print(f"Extracting images from {file_name}...")
                
                images_exp = []
                image_count = 1
                for page in doc.pages():
                    image_info = page.get_image_info(hashes=True, xrefs=True)
                    images = page.get_images()

                    # === Check if images have been extracted and extract images
                    # TODO ==== Replace with CV image detection ==== 
                    for image_meta_data in range(len(image_info)): 
                        if image_info[image_meta_data] not in images_exp: # check if image with same meta data has been added to exports 
                            images_exp.append(image_info[image_meta_data]) # add image meta data to exports 
                            image = doc.extract_image(images[image_meta_data][0]) # extract using specified xref 
                            imgout = open(f"{image_out}/image{image_count}.{image['ext']}", "wb") # image directory
                            imgout.write(image["image"]) # save image 
                            imgout.close()
                            image_count += 1
                print(f"Images Extracted from {file_name} and saved to {image_out}")

            if pathlib.Path(image_out).exists():
                if len(os.listdir(pathlib.Path(image_out))) > 0: 
                    print(f"{file_name} images already processed.")
                    print("Skipping...")
                else:
                    image_helper()
            else:
                image_helper()

    print("Extractions Complete!")


# === Usage ===
pdf_extract(file_paths, text_extraction=True, image_extraction=False)