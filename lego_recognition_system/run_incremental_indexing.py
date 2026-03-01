import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.build_reference_index import build_index

DATASET_DIR = os.path.join(PROJECT_ROOT, "render_local/32000")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/piezas_vectores")

if __name__ == "__main__":
    print(f"🚀 Iniciando indexación incremental para la pieza 32000...")
    # build_index automatically loads existing lego.index if found in OUTPUT_DIR
    build_index(DATASET_DIR, OUTPUT_DIR, unified=True)
    print("✅ Indexación completada.")
