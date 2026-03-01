"""
model_registry.py
Checks Google Drive and local models/ folder for existing trained models.
A part is considered "trained" if BOTH models/{ldraw_id}/best.pt AND models/{ldraw_id}/vector.pkl exist.
"""
import os
import logging

logger = logging.getLogger("LegoVision")

MODELS_FOLDER_NAME = "models"


def get_local_models_dir(project_root: str) -> str:
    return os.path.join(project_root, MODELS_FOLDER_NAME)


def check_local_model(project_root: str, ldraw_id: str) -> dict:
    """Check if a model exists in the local models/ directory (Strategy C)."""
    models_dir = get_local_models_dir(project_root)
    yolo_dir = os.path.join(models_dir, "yolo_model")
    piezas_dir = os.path.join(models_dir, "piezas_vectores")
    
    has_yolo = False
    if os.path.exists(yolo_dir):
        has_yolo = any(f.endswith('.pt') for f in os.listdir(yolo_dir))
        
    has_vector = False
    master_meta = os.path.join(piezas_dir, "lego_meta.pkl")
    if os.path.exists(master_meta):
        try:
            import pickle
            with open(master_meta, 'rb') as f:
                data = pickle.load(f)
                metadata = data.get('metadata', [])
                has_vector = any(m.get('ldraw_id') == str(ldraw_id) for m in metadata)
        except: pass
    
    # Check individual .index for legacy support
    if not has_vector:
        has_vector = os.path.exists(os.path.join(piezas_dir, f"{ldraw_id}.index"))
    
    # In Strategy C, it's 'complete' if we have the vector entry. YOLO is universal.
    return {"ldraw_id": ldraw_id, "has_yolo": has_yolo, "has_vector": has_vector, "is_complete": has_vector}


def check_drive_model(drive_service, drive_models_folder_id: str, ldraw_id: str) -> dict:
    """
    Check Google Drive for existing models for a part (Strategy C format).
    Looks inside drive_models_folder_id for yolo_model/*.pt and piezas_vectores/{ldraw_id}.pkl
    """
    has_yolo = False
    has_vector = False
    try:
        # Check YOLO folder
        q_yolo_dir = f"name='yolo_model' and '{drive_models_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        yolo_dirs = drive_service.files().list(q=q_yolo_dir, fields="files(id)").execute().get("files", [])
        if yolo_dirs:
            yolo_id = yolo_dirs[0]['id']
            q_pt = f"'{yolo_id}' in parents and name contains '.pt' and trashed=false"
            has_yolo = bool(drive_service.files().list(q=q_pt, fields="files(id)").execute().get("files", []))

        # Check Piezas folder
        q_piezas_dir = f"name='piezas_vectores' and '{drive_models_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        piezas_dirs = drive_service.files().list(q=q_piezas_dir, fields="files(id)").execute().get("files", [])
        if piezas_dirs:
            piezas_id = piezas_dirs[0]['id']
            q_pkl = f"name='{ldraw_id}.pkl' and '{piezas_id}' in parents and trashed=false"
            has_vector = bool(drive_service.files().list(q=q_pkl, fields="files(id)").execute().get("files", []))

    except Exception as e:
        logger.warning(f"⚠️ Drive check failed for {ldraw_id}: {e}")

    return {"ldraw_id": ldraw_id, "has_yolo": has_yolo, "has_vector": has_vector, "is_complete": has_vector}

def get_training_status(parts: list, project_root: str, drive_service=None, drive_models_folder_id: str = None) -> list:
    """
    For each part in the list, returns a status dict:
    {ldraw_id, name, has_yolo, has_vector, is_complete, source}
    Checks local first, then Drive if a service is provided.
    """
    results = []
    for part in parts:
        lid = part["ldraw_id"]
        local = check_local_model(project_root, lid)

        if local["is_complete"]:
            results.append({**part, **local, "source": "local"})
            continue

        if drive_service and drive_models_folder_id:
            drive = check_drive_model(drive_service, drive_models_folder_id, lid)
            if drive["is_complete"]:
                results.append({**part, **drive, "source": "drive"})
                continue

        # Not found anywhere
        results.append({**part, "has_yolo": local["has_yolo"], "has_vector": local["has_vector"], "is_complete": False, "source": "none"})

    return results


def filter_pending(status_list: list) -> list:
    """Returns only the parts that still need training."""
    return [s for s in status_list if not s["is_complete"]]
