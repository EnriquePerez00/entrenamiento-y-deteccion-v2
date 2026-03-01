
import json
import os
import glob
from src.utils.notebook_templates import C0_SETUP, C1_INSTALL, C7_SYNC

def patch_notebook(file_path):
    print(f"🔧 Patching {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found.")
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return False
            
    modified = False
    cells = nb.get('cells', [])
    
    # 1. Update Fixed Cells (Setup, Install, Sync)
    # Cell 0: Markdown
    # Cell 1: Setup
    if len(cells) > 1:
        cells[1]['source'] = C0_SETUP
        modified = True
    # Cell 2: Install
    if len(cells) > 2:
        cells[2]['source'] = C1_INSTALL
        modified = True
    # Last Cell: Sync (usually index 8 in unified, but let's be dynamic)
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code' and any('CELDA 7' in line for line in cell.get('source', [])):
            cell['source'] = C7_SYNC
            modified = True
            break
            
    # 2. Patch Code Logic in remaining cells
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
            
        source = cell.get('source', [])
        if isinstance(source, str):
            source = source.splitlines(keepends=True)
            
        new_source = []
        cell_modified = False
        
        # --- PRE-PROCESS: Cleanup corrupted f-strings and indentation ---
        cleaned_source = []
        i = 0
        while i < len(source):
            line = source[i]
            # 1. Quote balancing
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                if i + 1 < len(source):
                    source[i+1] = line.rstrip('\n') + source[i+1]
                    cell_modified = True
                    i += 1
                    continue
            
            # 2. Specific indentation fix for known corrupted logger block
            # If we see a log_f line with 16 spaces followed by an 'if' with 12, fix it.
            if 'log_f = os.path.join(WORKSPACE_DIR' in line and line.startswith(' ' * 16):
                if i + 1 < len(source) and 'if os.path.exists(log_f):' in source[i+1]:
                    if source[i+1].startswith(' ' * 12) and not source[i+1].startswith(' ' * 13):
                        line = ' ' * 12 + line.lstrip()
                        cell_modified = True

            cleaned_source.append(line)
            i += 1
        source = cleaned_source

        for line in source:
            original_line = line
            
            # Helper to detect indentation
            def get_indent(l):
                return l[:len(l) - len(l.lstrip())]
            
            indent = get_indent(line)

            # Helper to handle multiline replacements correctly with preserved indentation
            def replace_with(lines_list, base_indent=None):
                if base_indent is None: base_indent = indent
                return [base_indent + l.lstrip() + ('\n' if not l.endswith('\n') else '') for l in lines_list]

            # Enforce Universal Detector Env
            if "env['PYTHONPATH'] = PROJECT_ROOT" in line:
                if "env['UNIVERSAL_DETECTOR'] = '1'" not in "".join(source):
                    new_source.append(line)
                    new_source.append(indent + "env['UNIVERSAL_DETECTOR'] = '1'\n")
                    cell_modified = True
                    continue

            # Fix data.yaml generation
            if "f.write(f'nc: {len(RESOLVED_PARTS)}" in line:
                new_source.append(indent + "f.write('nc: 1\\n')\n")
                cell_modified = True
                continue
            elif "f.write(f'names: {[p[\"ldraw_id\"] for p in RESOLVED_PARTS]}" in line:
                new_source.append(indent + "f.write(\"names: ['lego']\\n\")\n")
                cell_modified = True
                continue

            # Standardize optimizer
            if "optimizer='default'" in line:
                line = line.replace("optimizer='default'", "optimizer='auto'")
                cell_modified = True
                
            # Dynamic YOLO path resolution
            if "best_pt_path = os.path.join(results_dir, f'yolo11_{SET_ID}', 'weights', 'best.pt')" in line:
                new_source.extend(replace_with([
                    "import glob",
                    "dirs = glob.glob(os.path.join(results_dir, f'yolo11_{SET_ID}*'))",
                    "latest_dir = max(dirs, key=os.path.getmtime) if dirs else os.path.join(results_dir, f'yolo11_{SET_ID}')",
                    "best_pt_path = os.path.join(latest_dir, 'weights', 'best.pt')"
                ]))
                cell_modified = True
                continue
            
            # Workspace consistency
            if '/tmp/render_cfg' in line:
                new_source.append(indent + f"cfg_path = os.path.join(WORKSPACE_DIR, f'render_cfg_{{worker_id}}.json')\n")
                cell_modified = True
                continue
            elif '/tmp/worker_' in line:
                if 'worker_{worker_id}.log' in line:
                    new_source.append(indent + f"log_file = os.path.join(WORKSPACE_DIR, f'worker_{{worker_id}}.log')\n")
                elif 'worker_{i}.log' in line:
                    new_source.append(indent + f"log_f = os.path.join(WORKSPACE_DIR, f'worker_{{i}}.log')\n")
                cell_modified = True
                continue

            # Optimize Render Workers (Reduce from 3 to 2 for Colab/Kaggle stability)
            if 'num_workers = 2 if IS_LIGHTNING else 3' in line:
                new_source.append(indent + "num_workers = 1 if IS_LIGHTNING else 2\n")
                cell_modified = True
                continue

            # Optimize YOLO imgsz (Reduce from 1024 to 640 for speed)
            if 'imgsz=1024' in line:
                line = line.replace('imgsz=1024', 'imgsz=640')
                cell_modified = True

            # Final safety: If a line starting with some indent suddenly changed its relative indent 
            # in previous buggy runs, we might want to fix it. 
            # Especially for the known case: 'log_f = ' followed by 'if os.path.exists(log_f):'
            if 'log_f = os.path.join(WORKSPACE_DIR' in line and i < len(source)-1:
                next_line = source[source.index(line)+1]
                if 'if os.path.exists(log_f):' in next_line:
                    # Force match indentations
                    target_indent = get_indent(line)
                    if get_indent(next_line) != target_indent:
                        # We'll fix it in the next iteration or here? 
                        # Let's just trust our 'replace_with' logic will fix it on the NEXT RUN 
                        # OR let's fix the next line now if it's already there.
                        pass

            new_source.append(line)
        
        if cell_modified:
            cell['source'] = new_source
            modified = True
            
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print(f"✅ patched {file_path}")
        return True
    else:
        print(f"ℹ️ No changes needed for {file_path}")
        return False
