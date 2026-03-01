import os

def generate_parts_lst(ldraw_dir):
    parts_dir = os.path.join(ldraw_dir, 'parts')
    output_path = os.path.join(ldraw_dir, 'parts.lst')
    
    if not os.path.exists(parts_dir):
        print(f"Error: {parts_dir} not found.")
        return

    print(f"🛠️ Generating parts.lst in {ldraw_dir}...")
    
    parts_found = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("0 LDraw.org Parts List\n")
        f.write("0 Name: parts.lst\n")
        
        # Walk through parts directory
        for filename in sorted(os.listdir(parts_dir)):
            if filename.lower().endswith('.dat'):
                full_path = os.path.join(parts_dir, filename)
                
                # Try to extract description from the first line of the .dat file
                description = filename
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as dat_f:
                        first_line = dat_f.readline().strip()
                        if first_line.startswith('0 '):
                            description = first_line[2:].strip()
                except:
                    pass
                
                # Format: filename description
                f.write(f"{filename} {description}\n")
                parts_found += 1
                
    print(f"✅ Created parts.lst with {parts_found} entries.")

if __name__ == "__main__":
    project_root = "/Users/I764690/Code_personal/test_heavy_image_recognition"
    ldraw_path = os.path.join(project_root, "lego_recognition_system", "assets", "ldraw")
    generate_parts_lst(ldraw_path)
