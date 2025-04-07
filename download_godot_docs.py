#!/usr/bin/env python3
import os
import requests
import zipfile
import shutil
from pathlib import Path
import sys

def main():
    # URL of the zip file to download
    url = "https://github.com/godotengine/godot-docs/archive/refs/heads/master.zip"
    
    # Path to save the downloaded zip file
    zip_path = "godot-docs-master.zip"
    
    # Path to the docs directory
    docs_dir = "docs"
    
    # Create docs directory if it doesn't exist
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f"Downloading {url}...")
    
    # Download the zip file using system proxy
    session = requests.Session()
    
    # Get proxy settings from environment or use system defaults
    print("Using system proxy settings...")
    
    # Download with proxy settings
    response = session.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Download completed. Extracting to {docs_dir}...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the name of the top-level directory in the zip
            top_dir = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall()
            
            # Move contents from the extracted directory to docs
            extracted_path = Path(top_dir)
            if extracted_path.exists():
                # Move all contents from extracted directory to docs
                for item in extracted_path.iterdir():
                    dest_path = Path(docs_dir) / item.name
                    if dest_path.exists():
                        if dest_path.is_dir():
                            shutil.rmtree(dest_path)
                        else:
                            os.remove(dest_path)
                    shutil.move(str(item), docs_dir)
                
                # Remove the now-empty extracted directory
                shutil.rmtree(top_dir)
        
        print("Extraction completed.")
        
        # Delete the zip file
        os.remove(zip_path)
        print(f"Deleted zip file: {zip_path}")
        
        print("Process completed successfully!")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

if __name__ == "__main__":
    main()
