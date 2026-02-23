
import sys
import os

print("Verifying imports...")
try:
    import torch
    import cv2
    import numpy
    import gradio
    import ultralytics
    import pandas
    import PIL
    print("Core libraries imported successfully.")
    import sam2
    from sam2.build_sam import build_sam2
    print("SAM 2 imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

print("Verifying local modules...")
try:
    from pipelines import StrategyA, StrategyB
    from utils import Benchmark
    print("Local modules imported successfully.")
except ImportError as e:
    print(f"Local Import Error: {e}")
    sys.exit(1)

print("Verifying Mock/Real SAM2 Logic...")
try:
    # Just instantiate to check logic
    s1 = StrategyA('yolo11n.pt', 'sam2_hiera_base.pt')
    s2 = StrategyB('yolo11n.pt', 'sam2_hiera_large.pt')
    print("Strategies instantiated successfully.")
except Exception as e:
    print(f"Strategy Instantiation Error: {e}")
    # Don't exit, might be model download issue which is expected on first run
    
print("Verification Complete.")
