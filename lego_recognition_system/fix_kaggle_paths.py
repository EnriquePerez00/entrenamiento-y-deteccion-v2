import json
import glob
import os

def fix_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code': continue
        source = cell['source']
        
        # Need to iterate backwards if we are inserting elements, but let's just use while loop
        i = 0
        while i < len(source):
            line = source[i]
            
            # Fix hardcoded /kaggle/working outputs that break in Colab
            if "results_dir = '/kaggle/working/results'" in line:
                source[i] = line.replace("'/kaggle/working/results'", "'/kaggle/working/results' if IS_KAGGLE else '/content/results'")
                changed = True
            elif "indices_dir = '/kaggle/working/indices'" in line:
                source[i] = line.replace("'/kaggle/working/indices'", "'/kaggle/working/indices' if IS_KAGGLE else '/content/indices'")
                changed = True
            elif "yolo_out = f'/kaggle/working/results/yolo11_{SET_ID}'" in line:
                source[i] = line.replace("f'/kaggle/working/results/yolo11_{SET_ID}'", "f'/kaggle/working/results/yolo11_{SET_ID}' if IS_KAGGLE else f'/content/results/yolo11_{SET_ID}'")
                changed = True
            elif "report_path = '/kaggle/working/performance_report.json'" in line:
                source[i] = line.replace("'/kaggle/working/performance_report.json'", "'/kaggle/working/performance_report.json' if IS_KAGGLE else '/content/performance_report.json'")
                changed = True
            elif "indices_src = '/kaggle/working/indices'" in line:
                source[i] = line.replace("'/kaggle/working/indices'", "'/kaggle/working/indices' if IS_KAGGLE else '/content/indices'")
                changed = True
            elif "yolo_results_zip = f'/kaggle/working/yolo_metricas_completas_{SET_ID}.zip'" in line:
                source[i] = line.replace("f'/kaggle/working/yolo_metricas_completas_{SET_ID}.zip'", "f'/kaggle/working/yolo_metricas_completas_{SET_ID}.zip' if IS_KAGGLE else f'/content/yolo_metricas_completas_{SET_ID}.zip'")
                changed = True
            elif "zip_path = f'/kaggle/working/lego_models_{SET_ID}.zip'" in line:
                source[i] = line.replace("f'/kaggle/working/lego_models_{SET_ID}.zip'", "f'/kaggle/working/lego_models_{SET_ID}.zip' if IS_KAGGLE else f'/content/lego_models_{SET_ID}.zip'")
                changed = True
            elif "blender_tarball = '/kaggle/working/blender.tar.xz'" in line:
                source[i] = line.replace("'/kaggle/working/blender.tar.xz'", "'/kaggle/working/blender.tar.xz' if IS_KAGGLE else '/content/blender.tar.xz'")
                changed = True
            
            # Fix kernel restore file extensions (.bin in kaggle vs .zip in colab/drive)
            elif "cycles_bin = os.path.join(kernel_dir, 'cycles_kernels.bin')" in line:
                if "cycles_bin = os.path.join(kernel_dir, 'cycles_kernels.zip')" not in source[i+1]:
                    source.insert(i+1, "        if not os.path.exists(cycles_bin): cycles_bin = os.path.join(kernel_dir, 'cycles_kernels.zip')\n")
                    changed = True
            elif "optix_bin = os.path.join(kernel_dir, 'optix_cache.bin')" in line:
                if "optix_bin = os.path.join(kernel_dir, 'optix_cache.zip')" not in source[i+1]:
                    source.insert(i+1, "        if not os.path.exists(optix_bin): optix_bin = os.path.join(kernel_dir, 'optix_cache.zip')\n")
                    changed = True
            
            # Fix the kaggle checking block in cell 7
            elif "PROJECT_ROOT = '/kaggle/working/lego_recognition_system'" in line:
                source[i] = line.replace("'/kaggle/working/lego_recognition_system'", "'/kaggle/working/lego_recognition_system' if IS_KAGGLE else '/content/drive/MyDrive/Lego_Training_75078'")
                changed = True
                
            i += 1
                
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f'Updated {filepath}')

for nb in glob.glob('*.ipynb'):
    fix_notebook(nb)
