import streamlit as st
import os
import sys

# Ensure src is in the python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gui.recognition_view import render_recognition_ui
from src.gui.launcher_view import render_launcher_ui

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
        st.sidebar.info(
            "**Two-Stage Architecture:**\n"
            "1. **YOLOv11** detects part bounding boxes.\n"
            "2. **Vector Search** classifies the cropped parts."
        )
        
        # Check if models exist
        yolo_model_path = os.path.join(PROJECT_ROOT, "training_results", "train", "weights", "best.pt")
        if not os.path.exists(yolo_model_path):
            yolo_model_path = os.path.join(PROJECT_ROOT, "yolo11n.pt")
            st.sidebar.warning("⚠️ Trained YOLO model not found. Using base yolo11n.pt for demonstration.")
            
        index_path = os.path.join(PROJECT_ROOT, "training_results", "75078-1_vector_index.pkl")
        if not os.path.exists(index_path):
            index_path = os.path.join(PROJECT_ROOT, "models", "75078-1_vector_index.pkl")
            if not os.path.exists(index_path):
                 st.sidebar.warning("⚠️ Vector Index not found. Classification results will be empty.")
                 index_path = None
        
        st.sidebar.subheader("Model Configuration")
        conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
        
        st.title("Interactive Part Recognition")
        st.markdown("Upload an image to test the 2-stage recognition pipeline.")
        
        uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            render_recognition_ui(uploaded_file, yolo_model_path, index_path, conf_threshold)
        else:
            test_img_path = os.path.join(PROJECT_ROOT, "temp_query.jpg")
            if os.path.exists(test_img_path):
                st.info("No image uploaded. Here is a sample query.")
                with open(test_img_path, "rb") as f:
                    render_recognition_ui(f, yolo_model_path, index_path, conf_threshold)
                    
    elif app_mode == "🚀 Entrenamiento (Launchpad)":
        render_launcher_ui(PROJECT_ROOT)

if __name__ == "__main__":
    main()
