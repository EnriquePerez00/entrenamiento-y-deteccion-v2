import json

notebook_path = "lego_recognition_system/master_unified_pipeline.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

source = nb['cells'][2]['source']
new_source = []

for line in source:
    if line.startswith("    BLENDER_PORTABLE_DIR"):
        new_source.append(line)
        continue
    
    # Check if inside our block that got indented too much
    if "BLENDER_PATH = os.path.join" in line or "if not os.path.exists(BLENDER_PATH):" in line or "Descargando Blender 4.2 LTS Portable" in line or "blender_tarball =" in line or "download_file(" in line or "'https://mirror" in line or "blender_tarball" in line or "Extrayendo Blender" in line or "dest = " in line or "subprocess.run" in line or "os.remove(blender_tarball)" in line or "os.environ['__EGL" in line or "os.environ['DISPLAY']" in line or "Blender portable listo:" in line:
        
        # If it has 8 spaces, strip 4
        if line.startswith("        ") and "download_file(" not in line and "blender_tarball" not in line and "'https://mirror" not in line: # Be careful with the nested function calls
            # Just do a blanket removal of 4 spaces if it starts with 8 spaces
             # Actual logic:
            pass
            
# Let's just do a simpler targeted replace
for i, line in enumerate(source):
    if line.startswith("        BLENDER_PATH"): source[i] = "    " + line[8:]
    if line.startswith("        if not os.path.exists(BLENDER_PATH):"): source[i] = "    " + line[8:]
    if line.startswith("            logger.info('Descargando Blender"): source[i] = "        " + line[12:]
    if line.startswith("            blender_tarball ="): source[i] = "        " + line[12:]
    if line.startswith("            download_file("): source[i] = "        " + line[12:]
    if line.startswith("                'https://mirror"): source[i] = "            " + line[16:]
    if line.startswith("                blender_tarball"): source[i] = "            " + line[16:]
    if line.startswith("            )"): source[i] = "        " + line[12:]
    if line.startswith("            logger.info('Extrayendo Blender"): source[i] = "        " + line[12:]
    if line.startswith("            dest ="): source[i] = "        " + line[12:]
    if line.startswith("            subprocess.run("): source[i] = "        " + line[12:]
    if line.startswith("            os.remove(blender_tarball)"): source[i] = "        " + line[12:]
    if line.startswith("        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']"): source[i] = "    " + line[8:]
    if line.startswith("        os.environ['DISPLAY']"): source[i] = "    " + line[8:]

nb['cells'][2]['source'] = source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)

print("Indentation fixed.")
