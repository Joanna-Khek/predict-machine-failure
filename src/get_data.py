import zipfile
import os
from pathlib import Path
import subprocess

def download_dataset(path_dir):
    
    # Download dataset using kaggle api
    command = f'kaggle competitions download -c playground-series-s3e17 -p {path_dir}'
    process = subprocess.Popen(command,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip())
            
    # Unzip file
    files = os.listdir(path_dir)
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(Path(path_dir, file), 'r') as zip_ref: 
                zip_ref.extractall(path_dir)
                print("Complete")

            # Remove unzipped file
            os.remove(Path(path_dir, file))


    