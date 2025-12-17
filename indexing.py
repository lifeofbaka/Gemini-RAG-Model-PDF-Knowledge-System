import json 
import glob 

pattern = "./Data/*/text/*.json"
files = glob.glob(pattern, recursive=True)


# === 