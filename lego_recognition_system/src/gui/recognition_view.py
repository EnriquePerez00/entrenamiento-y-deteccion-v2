import streamlit as st
import numpy as np
import os
import ssl
from PIL import Image, ImageDraw

# Fix macOS SSL certificate verification issue (common with Python from python.org)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex

@st.cache_resource(show_spinner="Loading AI Models...")
def load_models(models_dir):
    """Loads and caches all models found in the models directory."""
    if YOLO is None:
        st.error("The `ultralytics` package is required for YOLO inference.")
        return None, None, None
        
    # Search for YOLO model
    yolo_model = None
    yolo_dir = os.path.join(models_dir, "yolo_model")
    yolo_path = os.path.join(yolo_dir, "detector_universal.pt")
    
    if not os.path.exists(yolo_path) and os.path.exists(yolo_dir):
        pt_files = [f for f in os.listdir(yolo_dir) if f.endswith(".pt")]
        if pt_files:
            yolo_path = os.path.join(yolo_dir, sorted(pt_files)[-1])
            
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
        st.sidebar.info(f"🧠 Detector: `{os.path.basename(yolo_path)}`")
    else:
        st.warning(f"⚠️ No hay modelos YOLO (.pt) en {yolo_dir}")
    
    feature_extractor = FeatureExtractor(model_name='mobilenet_v3_small')
    
    # Load ALL available vector indices
    vector_index = VectorIndex()
    piezas_dir = os.path.join(models_dir, "piezas_vectores")
    index_files = []
    
    if os.path.exists(piezas_dir):
        index_files = [f for f in os.listdir(piezas_dir) if f.endswith(".pkl")]
        
    for idx_file in index_files:
        try:
            vector_index.load(os.path.join(piezas_dir, idx_file))
        except Exception as e:
            st.error(f"Error loading index {idx_file}: {e}")
            
    if not index_files:
        st.sidebar.warning("⚠️ No hay índices de vectores (.pkl)")
    else:
        st.sidebar.success(f"📚 {len(index_files)} Índices cargados ({len(vector_index.embeddings)} vectores)")
        
    return yolo_model, feature_extractor, vector_index


def render_recognition_ui(uploaded_file, models_dir, conf_threshold):
    """Renders the main recognition pipeline visualization."""
    
    try:
        # Load the image
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return
        
    yolo, extractor, v_index = load_models(models_dir)
    if yolo is None or extractor is None:
        return
        
    st.subheader("1. YOLO Detection (Phase 1)")
    
    with st.spinner("Running YOLO detector..."):
        # Run inference
        results = yolo.predict(image, conf=conf_threshold)
        
    if not results or len(results[0].boxes) == 0:
        st.info("No LEGO pieces detected in this image.")
        st.image(image, use_container_width=True)
        return
        
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    # Draw boxes for the UI
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = confidences[i]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), f"Part {i+1} ({conf:.2f})", fill="red")
        
    st.image(drawn_image, caption=f"Detected {len(boxes)} potential LEGO parts", use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. Vector Search Classification (Phase 2)")
    
    if v_index is None or not v_index.embeddings:
        st.warning("⚠️ No Vector Index loaded. Generating embeddings without classification.")
        
    # Process each detected bounding box
    identified_parts = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # Expand box slightly for better context
        pad = 5
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(image.width, x2+pad), min(image.height, y2+pad)
        
        cropped_img = image.crop((cx1, cy1, cx2, cy2))
        
        with st.expander(f"🧩 Part {i+1} - Confidence: {confidences[i]:.0%}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(cropped_img, caption="Cropped ROI", width=150)
                
            with col2:
                # Extract embedding
                embedding = extractor.get_embedding(cropped_img)
                
                if v_index is not None and v_index.embeddings:
                    matches = v_index.search(embedding, k=3)
                    
                    st.markdown("**Top Matches from Vector DB:**")
                    for match_idx, match in enumerate(matches):
                        ldraw_id = match['metadata'].get('ldraw_id', 'Unknown')
                        sim = match['similarity']
                        
                        # Count the best match for the final summary (threshold > 0.4 to be safe)
                        if match_idx == 0 and sim > 0.4:
                            identified_parts[ldraw_id] = identified_parts.get(ldraw_id, 0) + 1
                            
                        # Use a progress bar to show similarity
                        st.markdown(f"**{match_idx + 1}. Part ID: `{ldraw_id}`** (Sim: {sim:.2f})")
                        st.progress(max(0.0, min(1.0, float(sim))), text=None)
                else:
                    st.info(f"Generated Vector Embedding (Size: {embedding.shape[0]})")
                    st.code(str(embedding[:5]) + " ...")

    # Final Summary Metrics
    st.markdown("---")
    st.subheader("📊 Recognition Summary")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Total Bounding Boxes (YOLO)", len(boxes))
    col_b.metric("Total Pieces Identified (Vector DB)", sum(identified_parts.values()))
    
    if identified_parts:
        st.write("**Piece Breakdown:**")
        cols = st.columns(4)
        for idx, (p_id, count) in enumerate(sorted(identified_parts.items(), key=lambda item: item[1], reverse=True)):
            with cols[idx % 4]:
                st.info(f"🧱 **{p_id}**: {count} units")
