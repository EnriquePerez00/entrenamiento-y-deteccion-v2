import streamlit as st
import os
import json
from datetime import datetime
import shutil
from src.logic.part_resolver import resolve_set, resolve_piece
from src.logic.model_registry import get_training_status, filter_pending
from src.utils.drive_manager import DriveManager

def render_launcher_ui(project_root):
    st.header("🚀 LEGO Training Launchpad")
    st.markdown("Configura tu próximo entrenamiento para Kaggle/Colab.")

    # --- 1. Autenticación Drive ---
    st.subheader("1. Conexión con Google Drive")
    dm = None
    try:
        creds_path = os.path.join(project_root, "credentials.json")
        token_path = os.path.join(project_root, "token_1973.pickle")
        dm = DriveManager(credentials_path=creds_path, token_path=token_path)
        if dm.authenticate():
            st.success(f"✅ Conectado como: {dm.account_email}")
        else:
            st.error("❌ Fallo de autenticación en Drive.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo conectar a Drive: {e}. Se asumirá que todo falta.")

    st.divider()

    # --- 2. Selección de Objetivo ---
    st.subheader("2. Selección de Objetivo")
    
    mode = st.radio("Modo de entrenamiento:", ["Referencia Set", "Pieza Específica"], horizontal=True, key="training_mode")
    
    col1, _ = st.columns(2)
    with col1:
        if mode == "Referencia Set":
            target_id = st.text_input("Referencia Set (Batch ID):", value="75078-1", key="set_id_input")
            num_parts = st.slider("Máximo de piezas random:", 1, 50, 5, key="num_parts_slider")
        else:
            target_id = st.text_input("ID de la Pieza (LDraw):", value="2877", key="piece_id_input")
            num_parts = 1

    st.divider()

    # --- 2b. Motor de Renderizado ---
    st.subheader("3. Motor de Renderizado")
    st.info("🔬 **CYCLES via CUDA/OPTIX**: Motor por defecto para imágenes hiperrealistas (raytracing).")
    render_engine = "CYCLES"

    launch_date = datetime.now().strftime("%Y%m%d_%H%M")
    full_ref = f"{target_id}_{launch_date}"
    st.info(f"Referencia de sesión: **{full_ref}**")

    # Si el target_id cambia, limpiamos el estado previo para no confundir
    if 'prev_target_id' not in st.session_state or st.session_state['prev_target_id'] != target_id:
        if 'last_status' in st.session_state:
            del st.session_state['last_status']
        st.session_state['prev_target_id'] = target_id

    if st.button("🔍 Verificar Modelos en Drive"):
        if not dm or not dm.service:
            st.error("❌ Drive no autenticado. No se puede verificar.")
            return

        with st.spinner(f"Resolviendo piezas para {target_id}..."):
            try:
                if mode == "Referencia Set":
                    parts = resolve_set(target_id, max_parts=num_parts)
                else:
                    parts = resolve_piece(target_id)
                
                # Obtener Carpeta de Modelos en Drive
                parent_name = "Lego_Training_75078"
                root_id = dm.ensure_folder(parent_name)
                models_drive_id = dm.ensure_folder("models", parent_id=root_id)
                
                status = get_training_status(parts, project_root, drive_service=dm.service, drive_models_folder_id=models_drive_id)
                
                st.session_state['last_status'] = status
                st.session_state['current_full_ref'] = full_ref
                st.rerun() # Forzar refresco para mostrar la tabla
            except Exception as e:
                st.error(f"Error en verificación: {e}")

    # --- 3. Resultados ---
    if 'last_status' in st.session_state:
        status = st.session_state['last_status']
        st.subheader("3. Checklist de entrenamiento")
        
        pending = filter_pending(status)
        
        data_table = []
        for s in status:
            res = "✅ YA EXISTE" if s['is_complete'] else "❌ PENDIENTE"
            data_table.append({
                "LDraw ID": s['ldraw_id'], 
                "Nombre": s['name'], 
                "Estado": res, 
                "Ubicación": s.get('source', 'Drive/Local') if s['is_complete'] else "No encontrado"
            })
        
        st.table(data_table)

        if not pending:
            st.balloons()
            st.success("🎉 ¡Todas las piezas solicitadas ya están entrenadas! No es necesario generar archivos nuevos.")
            # Remove any previous session data to be safe
            if 'last_zip_ready' in st.session_state:
                del st.session_state['last_zip_ready']
        else:
            st.warning(f"Se detectaron **{len(pending)}** piezas para entrenar.")
            
            if st.button("📦 Generar ZIP Kaggle y Sincronizar"):
                with st.spinner("Preparando archivos para piezas faltantes..."):
                    try:
                        # Use a fresh timestamp for the actual generation to avoid stale names
                        session_ref = f"{target_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                        st.session_state['current_full_ref'] = session_ref
                        
                        # 1. Config (Solo con las piezas faltantes)
                        config_path = os.path.join(project_root, "config_train.json")
                        config_data = {
                            "session_reference": session_ref,
                            "target_parts": [p['ldraw_id'] for p in pending],
                            "render_settings": {"num_images": 300, "engine": render_engine}
                        }
                        with open(config_path, "w") as f:
                            json.dump(config_data, f, indent=4)
                        
                        # 2. Notebook
                        notebook_name = f"train_{session_ref}.ipynb"
                        notebook_path = os.path.join(project_root, notebook_name)
                        template_path = os.path.join(project_root, "master_unified_pipeline.ipynb")
                        if os.path.exists(template_path):
                            shutil.copy2(template_path, notebook_path)
                        
                        # 3. Sync & Pack
                        import sync_manager
                        # Append new files to global tracking without wiping the list
                        if config_path not in sync_manager.SYNC_FILES:
                            sync_manager.SYNC_FILES.append(config_path)
                        if notebook_path not in sync_manager.SYNC_FILES:
                            sync_manager.SYNC_FILES.append(notebook_path)
                            
                        # PROGRESSIVE YOLO TRAINING: Include latest weights in ZIP
                        models_dir = os.path.join(project_root, "models")
                        yolo_dir = os.path.join(models_dir, "yolo_model")
                        latest_model = None
                        
                        if os.path.exists(yolo_dir):
                            pt_files = [f for f in os.listdir(yolo_dir) if f.endswith(".pt")]
                            if pt_files:
                                # Prefer 'detector_universal.pt' or the latest by string sort
                                if "detector_universal.pt" in pt_files:
                                    latest_model = "detector_universal.pt"
                                else:
                                    latest_model = sorted(pt_files)[-1]
                                    
                        if latest_model:
                            model_path = os.path.join(yolo_dir, latest_model)
                            if model_path not in sync_manager.SYNC_FILES:
                                sync_manager.SYNC_FILES.append(model_path)
                            st.success(f"🧠 Adjuntando pesos base para entrenamiento progresivo: {latest_model}")
                        else:
                            st.warning("⚠️ No se encontraron pesos base locales en `models/yolo_model`. YOLO se entrenará desde cero (yolo11n-obb.pt).")
                        
                        zip_name = f"kaggle_{session_ref}.zip"
                        sync_manager.pack_for_kaggle(output_zip=os.path.join(project_root, zip_name))
                        sync_manager.sync_to_drive()
                        
                        st.session_state['last_zip_ready'] = {
                            "zip": zip_name,
                            "nb": notebook_name
                        }
                        st.success(f"✅ ¡Archivos generados para las {len(pending)} piezas pendientes!")

                    except Exception as e:
                        st.error(f"Fallo: {e}")

    # Instrucciones finales solo si hay algo generado
    if 'last_zip_ready' in st.session_state:
        res = st.session_state['last_zip_ready']
        st.markdown(f"""
        ---
        ### 📥 Listos para Kaggle
        1. Descarga/Localiza el ZIP: `{res['zip']}` (también subido a Drive).
        2. Súbelo a Kaggle y ejecuta el notebook: `{res['nb']}`.
        """)
