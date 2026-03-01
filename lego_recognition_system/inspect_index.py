import os
import pickle
import faiss
import numpy as np

def inspect_index(index_dir):
    print(f"🔍 Inspeccionando índice en: {index_dir}")
    
    meta_path = os.path.join(index_dir, "lego_meta.pkl")
    idx_path = os.path.join(index_dir, "lego.index")
    
    if not os.path.exists(meta_path) or not os.path.exists(idx_path):
        print("❌ Error: No se encontraron los archivos lego.index o lego_meta.pkl")
        return

    # 1. Inspect FAISS Index
    try:
        index = faiss.read_index(idx_path)
        print(f"📊 FAISS Index:")
        print(f"   - Total Vectores: {index.ntotal}")
        print(f"   - Dimensiones:    {index.d}")
        print(f"   - Métrica:        {'Inner Product (Similitud Coseno)' if isinstance(index, faiss.IndexFlatIP) else 'L2'}")
    except Exception as e:
        print(f"❌ Error leyendo lego.index: {e}")

    # 2. Inspect Metadata
    try:
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
            metadata = data.get('metadata', [])
            
        print(f"📋 Metadatos:")
        print(f"   - Total Entradas: {len(metadata)}")
        
        unique_ids = sorted(list(set([m.get('ldraw_id', 'Unknown') for m in metadata])))
        print(f"   - IDs Únicos ({len(unique_ids)}): {', '.join(unique_ids[:15])}{'...' if len(unique_ids) > 15 else ''}")
        
        if len(unique_ids) == 1:
            print(f"⚠️ ATENCIÓN: Solo hay UN tipo de pieza en el índice ({unique_ids[0]})")
        elif len(unique_ids) == 0:
            print(f"⚠️ ATENCIÓN: El índice está VACÍO")
            
    except Exception as e:
        print(f"❌ Error leyendo lego_meta.pkl: {e}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))
    target_dir = os.path.join(project_root, "models", "piezas_vectores")
    inspect_index(target_dir)
