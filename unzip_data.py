import zipfile
import os

zip_path = "./data/raw/dataset.zip"
extract_path = "./data/raw"

print("Starting to unzip your dataset... Please wait a moment.")

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✓ Done! Dataset unzipped successfully.")
else:
    print("Error: Could not find dataset.zip inside data/raw/")