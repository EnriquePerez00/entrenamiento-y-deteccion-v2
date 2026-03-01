import os
import datetime
from src.logic.trainer import ModelTrainer

def test_trainer_model_path():
    trainer = ModelTrainer(model_dir="./test_models")
    set_id = "test_set"
    
    # Create some dummy files to simulate timestamped models
    os.makedirs(trainer.model_dir, exist_ok=True)
    
    old_ts = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d_%H%M')
    new_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    
    old_model = os.path.join(trainer.model_dir, f"detector_set_{set_id}_{old_ts}.pt")
    new_model = os.path.join(trainer.model_dir, f"detector_set_{set_id}_{new_ts}.pt")
    
    with open(old_model, 'w') as f:
        f.write('dummy')
        
    with open(new_model, 'w') as f:
        f.write('dummy')
        
    # Test path fetching
    found_path = trainer.get_model_path(set_id)
    print(f"Old model created: {old_model}")
    print(f"New model created: {new_model}")
    print(f"Trainer found model: {found_path}")
    
    assert found_path == new_model, f"Expected {new_model}, but got {found_path}"
    print("✅ Logic works correctly.")
    
    # Cleanup
    os.remove(old_model)
    os.remove(new_model)
    os.rmdir(trainer.model_dir)

if __name__ == "__main__":
    test_trainer_model_path()
