import os
import sys

# Ensure src is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.notebook_patcher import patch_notebook

NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'master_unified_pipeline.ipynb')

def main():
    if patch_notebook(NOTEBOOK_PATH):
        print("Master notebook updated successfully!")
    else:
        print("No updates needed for master notebook.")

if __name__ == '__main__':
    main()
