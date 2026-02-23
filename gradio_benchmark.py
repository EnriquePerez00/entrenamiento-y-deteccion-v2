
import gradio as gr
import cv2
import pandas as pd
import numpy as np
from pipelines import StrategyA, StrategyB
from utils import draw_masks
import torch

# --- Initialize Pipelines ---
# Using placeholders for checkpoints. 
# Real application would ensure files exist or use 'yolo11n.pt' which ultralytics downloads automatically
print("Initializing Pipelines...")
try:
    pipeline_a = StrategyA(model_name='yolo11n.pt', sam_checkpoint='sam2_hiera_base.pt')
    pipeline_b = StrategyB(model_name='yolo11x.pt', sam_checkpoint='sam2_hiera_large.pt')
except Exception as e:
    print(f"Error initializing pipelines: {e}")
    pipeline_a = None
    pipeline_b = None

def safe_sum(d):
    total = 0.0
    for k, v in d.items():
        if isinstance(v, (int, float, np.number)):
            total += float(v)
    return round(total, 2)

def run_comparison(image, conf_thresh, iou_thresh, strat_a_model, use_tta, refine_iters):
    if image is None:
        return None, None, pd.DataFrame(), "Please upload an image."
    
    print(f"Processing image with shape: {image.shape}")
    
    # Resize if too large to prevent memory crashes on Mac
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"Resized image to: {image.shape}")

    log_text = "Starting Comparison...\n"
    metrics_data = []
    
    # --- Strategy A ---
    log_text += "Running Strategy A (Efficiency)...\n"
    # Update model if needed (simplified: initialized once, but could reload if users change dropdown)
    # For this prototype we stick to initialized models or we would need to reload in the Class.
    
    config_a = {'conf': conf_thresh, 'iou': iou_thresh}
    try:
        masks_a, boxes_a = pipeline_a.run(image, config_a)
        res_image_a = draw_masks(image, masks_a, boxes_a)
        
        m_a = pipeline_a.get_metrics()
        m_a['Strategy'] = 'A: Efficiency'
        m_a['Total Time (ms)'] = safe_sum(m_a)
        metrics_data.append(m_a)
        log_text += f"Strategy A Complete. {len(boxes_a)} objects detected.\n"
    except Exception as e:
        log_text += f"Strategy A Failed: {e}\n"
        res_image_a = image
        
    # --- Strategy B ---
    log_text += "Running Strategy B (Precision)...\n"
    config_b = {'conf': conf_thresh, 'iou': iou_thresh, 'tta': use_tta, 'refine_iters': refine_iters}
    try:
        masks_b, boxes_b = pipeline_b.run(image, config_b)
        res_image_b = draw_masks(image, masks_b, boxes_b)
        
        m_b = pipeline_b.get_metrics()
        m_b['Strategy'] = 'B: Precision'
        m_b['Total Time (ms)'] = safe_sum(m_b)
        metrics_data.append(m_b)
        log_text += f"Strategy B Complete. {len(boxes_b)} objects detected.\n"
    except Exception as e:
        log_text += f"Strategy B Failed: {e}\n"
        res_image_b = image

    # Finalize Logs and Table
    print(f"Metrics Data: {metrics_data}")
    df = pd.DataFrame(metrics_data)
    # Reorder columns to put Strategy first
    if not df.empty:
        cols = ['Strategy'] + [c for c in df.columns if c != 'Strategy']
        df = df[cols]
    
    return res_image_a, res_image_b, df, log_text

# --- GUI Layout ---
with gr.Blocks(title="LEGO Segmentation Benchmark") as demo:
    gr.Markdown("## 🧱 LEGO Sorter AI: Segmentation Strategy Benchmark")
    gr.Markdown("Compare **real-time efficiency** (YOLO11-S + SAM2-Base) vs **high-precision** (YOLO11-X + SAM2-Huge + Refinement).")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Header("Controls")
            input_image = gr.Image(label="Input Image", type="numpy")
            
            with gr.Accordion("Global Parameters", open=True):
                conf_slider = gr.Slider(0.0, 1.0, value=0.25, label="Confidence Threshold")
                iou_slider = gr.Slider(0.0, 1.0, value=0.45, label="IoU Threshold (NMS)")
            
            with gr.Group():
                gr.Markdown("### Strategy A Settings")
                gr.Markdown("*Current Model: YOLO11-Small (Fixed)*")
                # In full version, add dropdown to reload model
            
            with gr.Accordion("Strategy B Settings", open=True):
                tta_check = gr.Checkbox(value=True, label="Enable TTA (Test Time Augmentation)")
                refine_slider = gr.Slider(0, 3, value=1, step=1, label="Refinement Iterations")
                
            run_btn = gr.Button("RUN COMPARISON", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Header("Visual Results")
            with gr.Row():
                out_img_a = gr.Image(label="Strategy A: Efficiency")
                out_img_b = gr.Image(label="Strategy B: Precision")
            
            gr.Header("Performance Metrics")
            metrics_table = gr.Dataframe(label="Timing Breakdown (ms)")
            logs_box = gr.Textbox(label="Execution Logs", lines=5)

    run_btn.click(
        fn=run_comparison,
        inputs=[input_image, conf_slider, iou_slider, gr.State("yolo11n.pt"), tta_check, refine_slider], 
        outputs=[out_img_a, out_img_b, metrics_table, logs_box]
    )

if __name__ == "__main__":
    demo.launch(share=False)
