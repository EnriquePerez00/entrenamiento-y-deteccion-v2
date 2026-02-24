import json
import glob
import os

for nb_file in glob.glob('*.ipynb'):
    with open(nb_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if "PROJECT_ROOT = '/content/drive/MyDrive/Lego_Training'\\n" in line:
                    source[i] = line.replace("'Lego_Training'", "'Lego_Training_75078'")
                    changed = True
    
    if changed:
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f'Updated {nb_file}')
