import json
import glob
import os

def fix_notebook_mount(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code': continue
        source = cell['source']
        
        for i, line in enumerate(source):
            if "elif IS_COLAB:\n" == line or "elif IS_COLAB:" in line:
                # Check if next lines already have the mount command
                if i + 1 < len(source) and "drive.mount" not in source[i+1] and "drive.mount" not in "".join(source[i:i+5]):
                    source.insert(i+1, "    from google.colab import drive\n")
                    source.insert(i+2, "    drive.mount('/content/drive')\n")
                    changed = True
                    break
                
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f'Updated {filepath} with drive mount logic.')

for nb in glob.glob('*.ipynb'):
    fix_notebook_mount(nb)
