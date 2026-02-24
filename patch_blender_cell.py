import json

notebook_path = "patch_notebook.py"

with open(notebook_path, 'r') as f:
    lines = f.readlines()

out = []
in_c1 = False

for line in lines:
    if "c1_lines = [" in line:
        in_c1 = True
    
    if in_c1 and "    \"    if RENDER_ENGINE == 'EEVEE':\"," in line:
        out.append("    \"    BLENDER_PORTABLE_DIR = '/kaggle/working/blender-4.2.7-linux-x64' if IS_KAGGLE else '/content/blender-4.2.7-linux-x64'\",\n")
        out.append("    \"    BLENDER_PATH = os.path.join(BLENDER_PORTABLE_DIR, 'blender')\",\n")
        out.append("    \"    if not os.path.exists(BLENDER_PATH):\",\n")
        out.append("    \"        logger.info('Descargando Blender 4.2 LTS Portable (EGL)...')\",\n")
        out.append("    \"        blender_tarball = '/kaggle/working/blender.tar.xz' if IS_KAGGLE else '/content/blender.tar.xz'\",\n")
        out.append("    \"        download_file(\",\n")
        out.append("    \"            'https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.7-linux-x64.tar.xz',\",\n")
        out.append("    \"            blender_tarball\",\n")
        out.append("    \"        )\",\n")
        out.append("    \"        logger.info('Extrayendo Blender...')\",\n")
        out.append("    \"        dest = '/kaggle/working/' if IS_KAGGLE else '/content/'\",\n")
        out.append("    \"        subprocess.run(['tar', '-xf', blender_tarball, '-C', dest], check=True)\",\n")
        out.append("    \"        os.remove(blender_tarball)\",\n")
        out.append("    \"    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'\",\n")
        out.append("    \"    os.environ['DISPLAY'] = ''\",\n")
        out.append("    \"    logger.info(f'Blender portable listo: {BLENDER_PATH} | EGL OK')\",\n")
        
        # Skip everything until LDraw
        skip_mode = True
        continue
        
    if in_c1 and "    \"    if not os.path.exists('/tmp/ldraw'):\"," in line:
        skip_mode = False
        in_c1 = False # done patching this block

    if not in_c1 or not locals().get('skip_mode', False):
        out.append(line)

with open(notebook_path, 'w') as f:
    f.writelines(out)

print("patch_notebook.py modified.")
