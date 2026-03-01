"""
Notebook Generator v6.0 — Lightning AI Only (Hybrid Pipeline)
=============================================================
Generates a single Lightning AI notebook that expects
pre-rendered datasets from the local Mac M4 pipeline.
"""

import json
import os
import datetime
from src.utils.notebook_templates_v6 import (
    MARKDOWN_HEADER_V6,
    C0_SETUP_LIGHTNING,
    C1_LOAD_DATASET,
    C2_TRAIN_LIGHTNING,
    C3_INDEX,
    C4_SYNC_LIGHTNING,
)


def _make_code_cell(source_lines):
    """Creates a notebook code cell dict."""
    cleaned_lines = []
    for line in source_lines:
        # Remove literal \n strings that might have leaked from templates
        # and ensure one newline at the end of each line
        clean = line.replace('\\n', '').strip('\n\r')
        cleaned_lines.append(clean + '\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cleaned_lines
    }


def _make_markdown_cell(source_lines):
    """Creates a notebook markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }


def generate_lightning_v6(output_dir=None, timestamp=None):
    """
    Generates a Lightning AI v6.0 notebook (Hybrid Pipeline).
    
    Returns:
        Path to the generated notebook file.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'notebooks')
    
    lightning_dir = os.path.join(output_dir, 'lightning')
    os.makedirs(lightning_dir, exist_ok=True)
    
    # Build notebook structure (5 cells instead of 10)
    cells = [
        _make_markdown_cell(MARKDOWN_HEADER_V6),
        _make_code_cell(C0_SETUP_LIGHTNING),
        _make_code_cell(C1_LOAD_DATASET),
        _make_code_cell(C2_TRAIN_LIGHTNING),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    filename = "LightningAI_v6_Master.ipynb"
    filepath = os.path.join(lightning_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Generated: lightning/{filename}")
    return filepath
