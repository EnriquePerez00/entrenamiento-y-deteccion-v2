import shutil
import os
import sys

import zipfile

def zip_dataset(set_id, data_dir="./data/datasets"):
    """
    Zips the dataset for a given set_id to prepare for Google Drive upload.
    """
    source_dir = os.path.join(data_dir, set_id)
    if not os.path.exists(source_dir):
        print(f"Error: Dataset for set {set_id} not found at {source_dir}")
        return

    output_filename = os.path.join(data_dir, f"{set_id}") # shutil adds .zip
    print(f"Zipping {source_dir} to {output_filename}.zip ...")
    
    shutil.make_archive(output_filename, 'zip', source_dir)
    
    zip_path = output_filename + ".zip"
    
    # Validation
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        print(f"Error: Zip file creation failed or file is empty: {zip_path}")
        return None
        
    if not zipfile.is_zipfile(zip_path):
        print(f"Error: Created file is not a valid zip: {zip_path}")
        return None
        
    print(f"Success! Zip created at {zip_path} ({os.path.getsize(zip_path)} bytes)")
    return zip_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python drivesync.py <set_id>")
    else:
        zip_dataset(sys.argv[1])
