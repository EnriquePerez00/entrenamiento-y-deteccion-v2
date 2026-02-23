from ultralytics import YOLO
import os

class ModelTrainer:
    def __init__(self, model_dir="./models/yolo"):
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train_model(self, set_id, dataset_path, epochs=10, progress_callback=None):
        """
        Train a YOLO model on the generated dataset.
        dataset_path: Path to the data.yaml file or dataset directory.
        progress_callback: Optional function(progress_percent, status_string)
        """
        print(f"Initializing YOLO11n for Set {set_id}...")
        model = YOLO('yolo11n.pt') 
        
        # Callback for UI Progress
        if progress_callback:
            def on_train_epoch_end(trainer):
                current_epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                progress = int((current_epoch / total_epochs) * 100)
                
                # Try to get metrics
                # trainer.metrics is a dict, but structure varies. 
                # safer to just show epoch
                status = f"Training Epoch {current_epoch}/{total_epochs}"
                progress_callback(progress, status)
                
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

        project_name = f"yolo11_set_{set_id}"
        
        print(f"Starting training on device='mps' for {epochs} epochs...")
        
        # Train
        # 'data' arg usually expects a YAML file describing the dataset.
        # Ultralytics automatic download or local path.
        # We assume dataset_path is a properly formatted yaml file or dir with data.yaml
        if os.path.isdir(dataset_path):
             data_file = os.path.join(dataset_path, "data.yaml")
        else:
             data_file = dataset_path
             
        results = model.train(
            data=data_file,
            epochs=epochs,
            imgsz=640,
            device='mps', # Force Apple Silicon GPU
            project=self.model_dir,
            name=project_name,
            exist_ok=True,
            plots=True,
            # Optimization for Mac Pro
            workers=8,  # Parallel data loading
            batch=32,   # Improved throughput (adjust to 64 if memory allows)
            cache=True  # Cache images in RAM
        )
        
        print(f"Training complete. Best model saved in {self.model_dir}/{project_name}/weights/best.pt")
        return results

    def get_model_path(self, set_id):
        # Simplified path logic
        return os.path.join(self.model_dir, f"yolo11_set_{set_id}", "weights", "best.pt")

