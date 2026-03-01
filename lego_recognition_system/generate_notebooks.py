#!/usr/bin/env python3
"""
Generate Notebooks v5.0
========================
Generates environment-specific notebooks (Colab, Kaggle, LightningAI).
Replaces the old patch_unified_notebook.py approach.
"""

import os
import sys

# Ensure src is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.notebook_generator import generate_all


def main():
    print("🚀 Generating environment-specific notebooks...")
    notebooks_root = os.path.join(PROJECT_ROOT, 'notebooks')
    paths = generate_all(output_dir=notebooks_root)
    
    print(f"\n📋 Generated notebooks:")
    for env, path in paths.items():
        # Display the relpath from PROJECT_ROOT
        rel = os.path.relpath(path, PROJECT_ROOT)
        print(f"   • {env}: {rel}")
    
    print("\n✅ All notebooks generated successfully!")


if __name__ == '__main__':
    main()
