"""
Notebook Generator v5.0
========================
Generates environment-specific notebooks from templates.
Replaces the old notebook_patcher.py approach.
"""

import json
import os
import datetime
from src.utils.notebook_templates import (
    MARKDOWN_HEADER,
    get_c0_setup, C1_INSTALL, C2_RESOLVE, C3_VERIFY,
    get_c4_render, get_c4_5_filter, get_c5_train, C6_INDEX, get_c7_sync
)

ENV_LABELS = {
    'colab': 'Google Collab',
    'kaggle': 'Kaggle',
    'lightning': 'Lightning AI',
}

ENV_PREFIXES = {
    'colab': 'Colab',
    'kaggle': 'Kaggle',
    'lightning': 'LightningAI',
}

ENV_DIR_MAP = {
    'colab': 'collab',
    'kaggle': 'kaggle',
    'lightning': 'lightning',
}


def _make_code_cell(source_lines):
    """Creates a notebook code cell dict."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith('\n') else line + '\n' for line in source_lines]
    }


def _make_markdown_cell(source_lines, env):
    """Creates a notebook markdown cell dict with env label substitution."""
    label = ENV_LABELS.get(env, env)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line.replace('{env_label}', label) for line in source_lines]
    }


def generate_notebook(environment, output_dir=None, timestamp=None):
    """
    Generates an environment-specific notebook.
    
    Args:
        environment: 'colab', 'kaggle', or 'lightning'
        output_dir: Directory to write the notebook to (defaults to cwd)
        timestamp: Optional timestamp string (defaults to now)
    
    Returns:
        Path to the generated notebook file.
    """
    if environment not in ENV_LABELS:
        raise ValueError(f"Unknown environment: {environment}. Use: {list(ENV_LABELS.keys())}")
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    
    if output_dir is None:
        # Default to a 'notebooks' directory in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'notebooks')
    
    # Create environment-specific subdirectory
    env_subfolder = ENV_DIR_MAP.get(environment, environment)
    env_dir = os.path.join(output_dir, env_subfolder)
    os.makedirs(env_dir, exist_ok=True)
    
    # Build notebook structure
    cells = [
        _make_markdown_cell(MARKDOWN_HEADER, environment),
        _make_code_cell(get_c0_setup(environment)),
        _make_code_cell(C1_INSTALL),
        _make_code_cell(C2_RESOLVE),
        _make_code_cell(C3_VERIFY),
        _make_code_cell(get_c4_render(environment)),
        _make_code_cell(get_c4_5_filter(environment)),
        _make_code_cell(get_c5_train(environment)),
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
    
    # Generate filename: Colab_20260226_1545.ipynb
    prefix = ENV_PREFIXES[environment]
    filename = f"{prefix}_{timestamp}.ipynb"
    filepath = os.path.join(env_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Generated: {env_subfolder}/{filename}")
    return filepath


def generate_all(output_dir=None, timestamp=None):
    """Generates notebooks for all 3 environments."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    
    paths = {}
    for env in ['colab', 'kaggle', 'lightning']:
        paths[env] = generate_notebook(env, output_dir=output_dir, timestamp=timestamp)
    
    return paths
