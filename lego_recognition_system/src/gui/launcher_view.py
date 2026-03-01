import streamlit as st
import os
import sys
import json
from datetime import datetime
import shutil
from pathlib import Path
from src.logic.part_resolver import resolve_set, resolve_piece, update_universal_inventory
from src.logic.lego_colors import LEGO_COLORS
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
    
    mode = st.radio("Modo de entrenamiento:", ["Referencia Set", "Listado de piezas (separado por ,)", "Listado de Minifigs (separado por ,)"], horizontal=True, key="training_mode")
    
    col1, _ = st.columns(2)
    with col1:
        if mode == "Referencia Set":
            target_id = st.text_input("Referencia Set (Batch ID):", value="75078-1", key="set_id_input")
            num_parts = st.slider("Máximo de piezas random:", 1, 50, 5, key="num_parts_slider")
        elif mode == "Listado de piezas (separado por ,)":
            target_id = st.text_input("Listado de piezas (separado por ,):", value="2877, 3001", key="piece_id_input")
            num_parts = len([x.strip() for x in target_id.split(",") if x.strip()])
        else:
            target_id = st.text_input("Listado de Minifigs (separado por ,):", value="sw0001", key="minifig_id_input")
            num_parts = len([x.strip() for x in target_id.split(",") if x.strip()])

    st.divider()

    # --- 3. Motor de Renderizado ---
    st.subheader("3. Motor de Renderizado")
    
    # 🧩 Resolve pieces for detailed visualization
    detail_parts = []
    if mode == "Referencia Set" and target_id:
        try:
            resolved = resolve_set(target_id, max_parts=num_parts)
            detail_parts = resolved  # list of dicts with ldraw_id, color_id, color_name
        except: pass
    else:
        detail_parts = [{"ldraw_id": x.strip()} for x in target_id.split(",") if x.strip()]

    if detail_parts:
        # Import tiering logic from runner (replicated or imported)
        def get_tier(p_id):
            if p_id.startswith('sw') or len(p_id) > 5: return 'TIER_3'
            if p_id in ['32054', '3795']: return 'TIER_3'
            return 'TIER_2'
        
        tier_cfg = {
            'TIER_1': {'imgs': 30, 'res': 640, 'engine': 'EEVEE'},
            'TIER_2': {'imgs': 80, 'res': 1280, 'engine': 'EEVEE'},
            'TIER_3': {'imgs': 150, 'res': 2048, 'engine': 'CYCLES'},
        }

        table_data = []
        for part in detail_parts:
            p_id = part.get("ldraw_id", str(part))
            t = get_tier(p_id)
            cfg = tier_cfg.get(t, tier_cfg['TIER_2'])
            row = {
                "Pieza/Minifig": p_id,
                "Color": part.get("color_name", "-"),
                "Tier": t,
                "Imágenes": cfg['imgs'],
                "Resolución": f"{cfg['res']}px",
                "Motor Blender": cfg['engine']
            }
            table_data.append(row)
        
        st.table(table_data)
    else:
        st.info("Ingresa una referencia o ID de pieza para ver los detalles del motor.")

    st.info("🔬 **Sujeto a cambios**: Las minifiguras y TIER_3 siempre usan CYCLES para máximo detalle.")
    render_engine = "CYCLES"

    launch_date = datetime.now().strftime("%Y%m%d_%H%M")
    clean_id = target_id.replace(", ", "_").replace(",", "_")
    full_ref = f"{clean_id}_{launch_date}"
    st.info(f"Referencia de sesión: **{full_ref}**")

    # --- 4. Renderizado Local (Mac M4) ---
    st.divider()
    st.subheader("4. Renderizado Local (M4 Pro)")
    st.info("Genera imágenes locales con SSS y Raytracing optimizado para Apple Silicon.")
    
    if st.button("🍎 Iniciar Renderizado Local", type="primary"):
        # 1. Preparar config_train.json para el orquestador
        config_path = os.path.join(project_root, "config_train.json")
        
        if mode == "Referencia Set":
            try:
                parts = resolve_set(target_id, max_parts=num_parts)
                # Pass color_id from the set inventory
                all_requested_ids = [
                    {"part_id": p['ldraw_id'], "color_id": p.get('color_id', 15), "color_name": p.get('color_name', 'White')}
                    for p in parts
                ]
            except Exception as e:
                st.error(f"❌ Error al resolver el set: {e}")
                return
        elif mode == "Listado de piezas (separado por ,)":
            all_requested_ids = [x.strip() for x in target_id.split(",") if x.strip()]
            # Ask user to select a color for each piece
            piece_color_map = {}
            for pid in all_requested_ids:
                color_options = [f"{cid} - {info['name']}" for cid, info in LEGO_COLORS.items()]
                selected = st.selectbox(f"Color para pieza {pid}", options=color_options, key=f"color_{pid}")
                # extract numeric id
                selected_id = int(selected.split(" - ")[0])
                piece_color_map[pid] = selected_id
        else: # Minifigs
            all_requested_ids = [x.strip() for x in target_id.split(",") if x.strip()]

        if not all_requested_ids:
            st.warning("⚠️ No hay piezas seleccionadas para renderizar.")
            return

        # --- Global Duplication Check (Cache System) ---
        pending_ids = []
        skipped_ids = []
        render_base = os.path.join(project_root, "render_local")
        
        for p_id in all_requested_ids:
            # Normalize: p_id can be a string or a dict with 'part_id'
            if isinstance(p_id, dict):
                p_id_str = p_id.get("part_id", str(p_id))
            else:
                p_id_str = str(p_id)

            # Check if piece already has a folder with images
            piece_img_dir = os.path.join(render_base, p_id_str, "images")
            if os.path.exists(piece_img_dir):
                existing_imgs = [f for f in os.listdir(piece_img_dir) if f.lower().endswith('.jpg')]
                if len(existing_imgs) > 0:
                    skipped_ids.append(p_id_str)
                    continue

            # Build pending entry — always as dict with color_id
            if isinstance(p_id, dict):
                pending_ids.append(p_id)
            elif mode == "Listado de piezas (separado por ,)":
                color_id = piece_color_map.get(p_id_str, 15)
                pending_ids.append({"part_id": p_id_str, "color_id": color_id})
            else:
                pending_ids.append(p_id_str)
        
        if skipped_ids:
            st.info(f"⏭️ **Caché detectada:** Saltando {len(skipped_ids)} piezas que ya existen: `{', '.join(skipped_ids[:10])}{'...' if len(skipped_ids)>10 else ''}`")
            
        if not pending_ids:
            st.success("✅ Todas las piezas solicitadas ya están en la caché local.")
            # We don't return, we'll let it proceed to show samples and ZIP info
            # but we define a flag to skip the subprocess call
            skip_render = True
        else:
            skip_render = False

        if not skip_render:
            # Prepare config with ONLY missing pieces
            with open(config_path, "w") as f:
                json.dump({
                    "session_reference": full_ref,
                    "target_parts": pending_ids,
                    "render_settings": {"engine": "CYCLES"}
                }, f, indent=4)
            
            # --- UI Visualization Setup ---
            st.markdown("### ⚙️ Fase de Generación en curso")
            m1, m2, m3 = st.columns(3)
            total_images_ui = m1.metric("Total Imágenes", "Calculando...")
            progress_pct_ui = m2.metric("Progreso", "0%")
            status_ui = m3.metric("Estado", "Iniciando...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 2. Ejecutar run_local_render.py
            import subprocess
            render_script = os.path.join(project_root, "run_local_render.py")
            cmd = [sys.executable, render_script]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            total_found = 0
            for line in process.stdout:
                # Parse total plan
                if "Total Render Plan:" in line:
                    try:
                        total_found = int(line.split("Total Render Plan:")[1].strip().split(" ")[0])
                        total_images_ui.metric("Total Imágenes", f"{total_found}")
                        status_ui.metric("Estado", "Renderizando...")
                    except: pass
                    
                if "Progress:" in line:
                    try:
                        parts_progress = line.split("Progress:")[1].strip().split(" ")[0]
                        done, total = map(int, parts_progress.split("/"))
                        pct = (done / total)
                        progress_bar.progress(min(1.0, pct))
                        progress_pct_ui.metric("Progreso", f"{int(pct*100)}%")
                        status_text.text(f"🚀 Renderizando imagen {done} de {total}...")
                    except:
                        pass
                else:
                    l_strip = line.strip()
                    if l_strip: status_text.text(l_strip)
            
            process.wait()
            render_success = (process.returncode == 0)
        else:
            render_success = True # Skipped because of cache

        if render_success:
            if not skip_render:
                status_ui.metric("Estado", "Finalizado ✅")
            st.success("✅ Renderizado local completado.")
            render_dir = os.path.join(project_root, "render_local")
            all_images = list(Path(render_dir).rglob("images/*.jpg"))
            if all_images:
                import random
                st.markdown("### 🖼️ Muestras Aleatorias (Caché Local)")
                
                # Use session state to allow refreshing samples without re-rendering
                if st.button("🔄 Refrescar Muestras"):
                    st.rerun()

                cols = st.columns(4)
                num_to_show = min(4, len(all_images))
                samples = random.sample(all_images, num_to_show)
                for i, img_p in enumerate(samples):
                    # Get piece ID from parent directory name
                    piece_id = img_p.parent.parent.name
                    cols[i].image(str(img_p), caption=f"Pieza: {piece_id}", use_container_width=True)
            
            # Show ZIP if generated
            import glob as _glob
            zips = sorted(_glob.glob(os.path.join(project_root, "lightning_dataset_*.zip")))
            if zips:
                latest_zip = zips[-1]
                zip_size = os.path.getsize(latest_zip) / (1024 * 1024)
                st.success(f"📦 ZIP listo: **{os.path.basename(latest_zip)}** ({zip_size:.1f} MB)")
                st.session_state['latest_lightning_zip'] = latest_zip
        else:
            st.error(f"❌ Fallo {process.returncode}")

    # --- 5. Preparar Lightning AI Package ---
    st.divider()
    st.subheader("5. Preparar para Lightning AI")
    
    # Detect existing ZIPs
    import glob as _glob
    existing_zips = sorted(_glob.glob(os.path.join(project_root, "lightning_dataset_*.zip")))
    
    if existing_zips:
        latest_zip = existing_zips[-1]
        zip_size = os.path.getsize(latest_zip) / (1024 * 1024)
        st.info(f"📦 Último dataset: **{os.path.basename(latest_zip)}** ({zip_size:.1f} MB)")
        
        # Show manifest if available
        manifest_path = os.path.join(project_root, "render_local", "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            st.markdown(f"**{manifest.get('total_images', '?')}** imágenes | "
                       f"**{manifest.get('total_pieces', '?')}** piezas | "
                       f"**{manifest.get('duplicates_removed', 0)}** duplicados eliminados")
    else:
        st.warning("⚠️ No hay dataset generado. Usa el botón de renderizado local primero.")
    
    if st.button("⚡ Generar Notebook Lightning AI v6.0"):
        try:
            from src.utils.notebook_generator_v6 import generate_lightning_v6
            gen_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            nb_path = generate_lightning_v6(
                output_dir=os.path.join(project_root, "notebooks"),
                timestamp=gen_timestamp
            )
            st.success(f"✅ Notebook generado: **{os.path.basename(nb_path)}**")
            st.markdown(f"""
**📋 Pasos siguientes:**
1. Sube `{os.path.basename(existing_zips[-1]) if existing_zips else 'lightning_dataset_*.zip'}` a Lightning AI
2. Sube `{os.path.basename(nb_path)}` a Lightning AI
3. Ejecuta el notebook en el Studio con GPU
""")
        except Exception as e:
            st.error(f"❌ Error generando notebook: {e}")

    # Limpiar estado si cambia el target_id (opcional, mantener minimalista)
    if 'prev_target_id' not in st.session_state or st.session_state['prev_target_id'] != target_id:
        st.session_state['prev_target_id'] = target_id

