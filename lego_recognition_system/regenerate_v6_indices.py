import os
import sys
import shutil
import zipfile
import glob
from pathlib import Path

def setup_paths():
    """Detects PROJECT_ROOT and WORKSPACE_DIR based on v6 notebook logic."""
    # 1. Detect PROJECT_ROOT (searching for 'src' folder)
    root = os.getcwd()
    while root != '/' and not os.path.exists(os.path.join(root, 'src')):
        root = os.path.dirname(root)
    
    if not os.path.exists(os.path.join(root, 'src')):
        root = os.getcwd() # Fallback

    # 2. Workspace logic (matches Celda 0 of v6)
    workspace = os.path.join(os.getcwd(), 'lightning_workspace')
    dataset_dir = os.path.join(workspace, 'datasets')
    
    # Try to find the set_id (typically 'lego_v6' in v6)
    set_id = 'lego_v6'
    
    # If there's a config_train.json, try to extract the real set_id
    config_path = os.path.join(root, 'config_train.json')
    if os.path.exists(config_path):
        import json
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                ref = cfg.get('session_reference', 'lego_v6')
                set_id = ref.split('_')[0]
        except: pass

    dataset_path = os.path.join(dataset_dir, set_id)
    indices_dir = os.path.join(root, 'models', 'piezas_vectores')
    
    return root, dataset_path, indices_dir

def main():
    print("🛰️ Iniciando Re-generación de Índices (Modo Remoto/Notebook)...")
    
    project_root, dataset_path, indices_dir = setup_paths()
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.logic.build_reference_index import build_index

    print(f"📍 Project Root: {project_root}")
    print(f"📊 Dataset Path: {dataset_path}")
    print(f"📂 Output Dir:   {indices_dir}")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: No se encontró el dataset unificado en {dataset_path}")
        print("Asegúrate de haber ejecutado la Celda 1 del notebook v6.")
        return

    # 1. Generar los índices
    os.makedirs(indices_dir, exist_ok=True)
    print("🧠 Ejecutando build_index...")
    try:
        build_index(dataset_path, indices_dir, unified=True)
        print("✅ Índices generados correctamente.")
    except Exception as e:
        print(f"❌ Error durante la generación: {e}")
        return

    # 2. Crear ZIP de resultados en la raíz
    zip_path = os.path.join(project_root, "indices.zip")
    files_to_zip = ["lego.index", "lego_meta.pkl"]
    
    print(f"🗜️ Comprimiendo resultados en {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        count = 0
        for f_name in files_to_zip:
            f_path = os.path.join(indices_dir, f_name)
            if os.path.exists(f_path):
                z.write(f_path, f_name)
                print(f"   + Agregado: {f_name}")
                count += 1
        
        if count == 0:
            print("⚠️ No se encontró ningún archivo para comprimir.")
            return

    print(f"\n🎉 ¡Proceso completado! Archivo listo para descarga: {zip_path}")

if __name__ == "__main__":
    main()
