import zipfile
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
INDEX_DIR = os.path.join(PROJECT_ROOT, "models/piezas_vectores")
ZIP_PATH = os.path.join(PROJECT_ROOT, "indices.zip")

def create_indices_zip():
    print(f"📦 Creando {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(INDEX_DIR):
            for file in files:
                if file.endswith(('.index', '.pkl', '.json')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, INDEX_DIR)
                    zipf.write(file_path, arcname=arcname)
                    print(f"  📎 Añadido: {arcname}")

if __name__ == "__main__":
    create_indices_zip()
