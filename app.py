# Selective imports to keep startup light
import streamlit as st
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.join(ROOT, "lego_recognition_system")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="LEGO Vision AI & Trainer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("🧩 LEGO Vision AI")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.selectbox("Selecciona aplicación:", ["🚀 Entrenamiento (Launchpad)", "🔍 Reconocimiento (Test)"])
    
    st.sidebar.markdown("---")
    
    if app_mode == "🔍 Reconocimiento (Test)":
        from src.gui.recognition_view import render_recognition_ui
        from src.utils.drive_manager import DriveManager
        st.sidebar.info(
            "**Two-Stage Architecture:**\n"
            "1. **YOLOv11** detects part bounding boxes.\n"
            "2. **Vector Search** classifies the cropped parts."
        )

        # ── Drive Sync ──────────────────────────────────────────────────
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)

        with st.sidebar.expander("☁️ Sincronización Drive", expanded=True):
            try:
                creds_path = os.path.join(PROJECT_ROOT, "credentials.json")
                token_path = os.path.join(PROJECT_ROOT, "token_1973.pickle")
                dm = DriveManager(credentials_path=creds_path, token_path=token_path)
                if dm.authenticate():
                    st.write(f"✅ Drive: `{dm.account_email}`")

                    # Locate the models folders on Drive
                    root_folder_id  = dm.ensure_folder("Lego_Training_75078")
                    models_folder_id = dm.ensure_folder("models", parent_id=root_folder_id)
                    yolo_folder_id = dm.ensure_folder("yolo_model", parent_id=models_folder_id)
                    piezas_folder_id = dm.ensure_folder("piezas_vectores", parent_id=models_folder_id)
                    
                    # Local folders
                    local_yolo_dir = os.path.join(models_dir, "yolo_model")
                    local_piezas_dir = os.path.join(models_dir, "piezas_vectores")
                    os.makedirs(local_yolo_dir, exist_ok=True)
                    os.makedirs(local_piezas_dir, exist_ok=True)

                    # Get Drive files
                    # We need modifiedTime to check which PT is newer
                    query_yolo = f"'{yolo_folder_id}' in parents and trashed=false and name contains '.pt'"
                    yolo_files = dm.service.files().list(q=query_yolo, fields="files(id, name, modifiedTime)").execute().get('files', [])
                    
                    query_piezas = f"'{piezas_folder_id}' in parents and trashed=false and name contains '.pkl'"
                    piezas_files = dm.service.files().list(q=query_piezas, fields="files(id, name)").execute().get('files', [])

                    to_download = []
                    
                    # YOLO Logic: Download if missing, or if Drive has a newer one
                    import dateutil.parser
                    import datetime
                    import pytz
                    
                    for f in yolo_files:
                        local_path = os.path.join(local_yolo_dir, f['name'])
                        if not os.path.exists(local_path):
                            to_download.append((f, local_path))
                        else:
                            remote_mtime = dateutil.parser.parse(f['modifiedTime'])
                            local_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(local_path), pytz.UTC)
                            if remote_mtime > local_mtime:
                                to_download.append((f, local_path))

                    # PKL Logic: Download only missing ones
                    for f in piezas_files:
                        local_path = os.path.join(local_piezas_dir, f['name'])
                        if not os.path.exists(local_path):
                            to_download.append((f, local_path))

                    if to_download:
                        st.write(f"📥 Descargando {len(to_download)} modelo(s) nuevos/actualizados...")
                        prog = st.progress(0)
                        for idx, (df, local_path) in enumerate(to_download):
                            st.write(f"  ↓ `{df['name']}`")
                            dm.download_file(df['id'], local_path)
                            prog.progress((idx + 1) / len(to_download))
                        st.success("✅ Modelos sincronizados.")
                    else:
                        st.write(f"✅ Local al día ({len(piezas_files)} piezas, {len(yolo_files)} YOLO).")

                else:
                    st.warning("⚠️ No se pudo autenticar Drive.")
            except Exception as e:
                st.warning(f"⚠️ Drive no disponible: {e}")

        # ── Recognition UI ──────────────────────────────────────────────
        st.title("Interactive Part Recognition")
        st.markdown("Sube una imagen para probar el reconocimiento de **2 etapas (Estrategia C)**.")

        # Add configuration to sidebar
        st.sidebar.markdown("### Configuración")
        conf_threshold = st.sidebar.slider("Confianza YOLO (Umbral)", 0.1, 1.0, 0.4, 0.05)

        uploaded_file = st.file_uploader("Elige una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            render_recognition_ui(uploaded_file, models_dir, conf_threshold)
        else:
            test_img_path = os.path.join(PROJECT_ROOT, "temp_query.jpg")
            if os.path.exists(test_img_path):
                st.info("Mostrando imagen de muestra.")
                with open(test_img_path, "rb") as f:
                    render_recognition_ui(f, models_dir, conf_threshold)

    elif app_mode == "🚀 Entrenamiento (Launchpad)":
        from src.gui.launcher_view import render_launcher_ui
        render_launcher_ui(PROJECT_ROOT)

if __name__ == "__main__":
    main()
