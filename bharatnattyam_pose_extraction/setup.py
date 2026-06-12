import zipfile
import os

zip_path = 'archive.zip'
extract_to = 'data/raw/'

#Ensure the directory exists
os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extract_to)
print("Extraction complete")