import re

with open('patch_notebook.py', 'r') as f:
    content = f.read()

# Replace the specific block of lines
old_block = """    "    BLENDER_PORTABLE_DIR = '/kaggle/working/blender-4.2.7-linux-x64' if IS_KAGGLE else '/content/blender-4.2.7-linux-x64'",
    "    BLENDER_PATH = os.path.join(BLENDER_PORTABLE_DIR, 'blender')",
    "    if not os.path.exists(BLENDER_PATH):",
    "        logger.info('Descargando Blender 4.2 LTS Portable (EGL)...')",
    "        blender_tarball = '/kaggle/working/blender.tar.xz' if IS_KAGGLE else '/content/blender.tar.xz'",
    "        download_file(",
    "            'https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.7-linux-x64.tar.xz',",
    "            blender_tarball",
    "        )",
    "        logger.info('Extrayendo Blender...')",
    "        dest = '/kaggle/working/' if IS_KAGGLE else '/content/'",
    "        subprocess.run(['tar', '-xf', blender_tarball, '-C', dest], check=True)",
    "        os.remove(blender_tarball)",
    "    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'",
    "    os.environ['DISPLAY'] = ''",
    "    logger.info(f'Blender portable listo: {BLENDER_PATH} | EGL OK')",
"""

new_block = """    "    BLENDER_PORTABLE_DIR = '/kaggle/working/blender-4.2.7-linux-x64' if IS_KAGGLE else '/content/blender-4.2.7-linux-x64'",
    "    BLENDER_PATH = os.path.join(BLENDER_PORTABLE_DIR, 'blender')",
    "    if not os.path.exists(BLENDER_PATH):",
    "        logger.info('Descargando Blender 4.2 LTS Portable (EGL)...')",
    "        blender_tarball = '/kaggle/working/blender.tar.xz' if IS_KAGGLE else '/content/blender.tar.xz'",
    "        download_file(",
    "            'https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.7-linux-x64.tar.xz',",
    "            blender_tarball",
    "        )",
    "        logger.info('Extrayendo Blender...')",
    "        dest = '/kaggle/working/' if IS_KAGGLE else '/content/'",
    "        subprocess.run(['tar', '-xf', blender_tarball, '-C', dest], check=True)",
    "        os.remove(blender_tarball)",
    "    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'",
    "    os.environ['DISPLAY'] = ''",
    "    logger.info(f'Blender portable listo: {BLENDER_PATH} | EGL OK')",
"""

content = content.replace(old_block, new_block)

# Regenerate master notebook from patch_notebook
with open('patch_notebook.py', 'w') as f:
    f.write(content)

import subprocess
subprocess.run(['python3', 'patch_notebook.py'])
