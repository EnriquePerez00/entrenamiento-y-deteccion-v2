from src.logic.trainer import ModelTrainer
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/yolo_model/lightning_workspace_results_yolo_lego_v6_weights_best.pt")
DATASET_DIR = os.path.join(PROJECT_ROOT, "render_local/32000")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models/yolo_model")

def train_incremental():
    print(f"🚀 Iniciando entrenamiento incremental...")
    trainer = ModelTrainer(model_dir=MODELS_DIR)
    
    trainer.train_model(
        set_id="32000",
        dataset_path=DATASET_DIR,
        epochs=5,
        base_model=MODEL_PATH
    )

if __name__ == "__main__":
    train_incremental()
