import json
import os

notebook_path = "lego_recognition_system/master_unified_pipeline.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Define cells as lists of strings WITHOUT the trailers
# We will add the real newline character \n to each one.

c4_raw = [
    "# " + "="*69,
    "# CELDA 4: Generacion de Dataset (Multi-GPU High Detail)",
    "# " + "="*69,
    "import subprocess, concurrent.futures, torch, time, psutil",
    "",
    "total_images_to_render = LAUNCH_CONFIG.get('render_settings', {}).get('num_images', 200) if LAUNCH_CONFIG else 200",
    "NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1",
    "num_workers = NUM_GPUS * 2 if RENDER_ENGINE == 'CYCLES' else NUM_GPUS",
    "images_per_worker = max(1, total_images_to_render // num_workers)",
    "logger.info(f'GPUs: {NUM_GPUS} | Workers: {num_workers} | Imgs/worker: {images_per_worker}')",
    "",
    "def run_render_worker(worker_id):",
    "    gpu_id = worker_id % NUM_GPUS",
    "    worker_cfg = {",
    "        'set_id': SET_ID, 'parts': RESOLVED_PARTS, 'num_images': images_per_worker,",
    "        'offset_idx': worker_id * images_per_worker, 'output_base': os.path.join(DATASET_DIR, SET_ID),",
    "        'assets_dir': os.path.join(PROJECT_ROOT, 'assets'), 'ldraw_path': LDRAW_PATH,",
    "        'addon_path': ADDON_PATH, 'render_engine': RENDER_ENGINE",
    "    }",
    "    cfg_path = f'/tmp/render_cfg_{worker_id}.json'",
    "    with open(cfg_path, 'w') as f: json.dump(worker_cfg, f)",
    "    env = os.environ.copy()",
    "    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)",
    "    env['PYTHONPATH'] = PROJECT_ROOT",
    "    log_file = f'/tmp/worker_{worker_id}.log'",
    "    with open(log_file, 'w') as f_out:",
    "        subprocess.run([BLENDER_PATH, '--background', '--python',",
    "            os.path.join(PROJECT_ROOT, 'src', 'blender_scripts', 'scene_setup.py'),",
    "            '--', cfg_path], stdout=f_out, stderr=subprocess.STDOUT, env=env)",
    "",
    "images_dir = os.path.join(DATASET_DIR, SET_ID, 'images')",
    "os.makedirs(images_dir, exist_ok=True)",
    "",
    "with timer.step('Blender Render Workers'):",
    "    print('\\n🚀 INICIANDO RENDER | ' + str(num_workers) + ' Workers | ' + RENDER_ENGINE + ' | Total: ' + str(total_images_to_render))",
    "    print('=' * 100)",
    "    start_t = time.time()",
    "    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:",
    "        futures = [executor.submit(run_render_worker, i) for i in range(num_workers)]",
    "        while any(f.running() for f in futures):",
    "            curr_imgs = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0",
    "            elapsed = time.time() - start_t",
    "            worker_status = []",
    "            for i in range(num_workers):",
    "                tail = 'Iniciando...' ",
    "                log_f = f'/tmp/worker_{i}.log'",
    "                if os.path.exists(log_f):",
    "                    try:",
    "                        with open(log_f, 'r') as f:",
    "                            last_line = f.readlines()[-1].strip()[-40:]",
    "                            tail = last_line",
    "                    except: pass",
    "                worker_status.append(f'W{i}: {tail}')",
    "            cpu, ram = psutil.cpu_percent(), psutil.virtual_memory().percent",
    "            vram = torch.cuda.memory_reserved(0)/1e9 if torch.cuda.is_available() else 0",
    "            fps = curr_imgs/elapsed if elapsed > 0 else 0",
    "            pct = (curr_imgs/total_images_to_render)*100",
    "            print(f'[{pct:>3.0f}%] {curr_imgs}/{total_images_to_render} | {fps:.1f} FPS | CPU {cpu}% | RAM {ram}% | VRAM {vram:.1f}GB | {int(elapsed)}s')",
    "            for s in worker_status: print(f'  ↳ {s}')",
    "            print('\\033[F' * (num_workers + 2), end='', flush=True)",
    "            time.sleep(3)",
    "    print('\\n' * (num_workers + 1) + '=' * 100)",
    "    print(f'✅ Render completado: {len(os.listdir(images_dir))} imágenes.')"
]

c5_raw = [
    "# " + "="*69,
    "# CELDA 5: YOLO Training DDP (v3.16 High Detail)",
    "# " + "="*69,
    "from ultralytics import YOLO",
    "import torch, time as _time",
    "",
    "if not PARTS_TO_TRAIN:",
    "    print('No hay piezas pendientes. Saltando entrenamiento.')",
    "else:",
    "    dataset_path = os.path.join(DATASET_DIR, SET_ID)",
    "    os.makedirs(dataset_path, exist_ok=True)",
    "    data_yaml = os.path.join(dataset_path, 'data.yaml')",
    "    with open(data_yaml, 'w') as f:",
    "        f.write(f'path: {os.path.abspath(dataset_path)}\\n')",
    "        f.write('train: images\\nval: images\\nnc: 1\\nnames: [LEGO_PART]\\n')",
    "",
    "    results_dir = '/kaggle/working/results' if os.path.exists('/kaggle') else os.path.join(PROJECT_ROOT, 'results')",
    "    models_out  = '/kaggle/working/models'  if os.path.exists('/kaggle') else os.path.join(PROJECT_ROOT, 'models')",
    "    model = YOLO('yolo11n.pt')",
    "",
    "    num_gpus = torch.cuda.device_count()",
    "    train_device = list(range(num_gpus)) if num_gpus > 1 else (0 if num_gpus == 1 else 'cpu')",
    "    train_batch = (32 * num_gpus * 4) if num_gpus > 1 else 128",
    "",
    "    _t_epoch = [_time.time()]",
    "    def _on_epoch_end(trainer):",
    "        elapsed = _time.time() - _t_epoch[0]",
    "        m = trainer.metrics or {}",
    "        vram = torch.cuda.memory_reserved(0)/1e9 if torch.cuda.is_available() else 0",
    "        print(f'\\n[EPOCH {trainer.epoch+1}/{trainer.epochs}] Time: {elapsed:.1f}s | Box: {m.get(\"train/box_loss\",0):.4f} | Cls: {m.get(\"train/cls_loss\",0):.4f} | mAP50: {m.get(\"metrics/mAP50(B)\",0):.4f} | VRAM: {vram:.1f}GB', flush=True)",
    "        _t_epoch[0] = _time.time()",
    "    model.add_callback('on_train_epoch_end', _on_epoch_end)",
    "",
    "    print(f'\\n🚀 YOLO DDP | GPUs: {num_gpus} | batch={train_batch} | cache=RAM')",
    "    print('=' * 100)",
    "    with timer.step('YOLO Training'): ",
    "        model.train(data=data_yaml, epochs=50, imgsz=640, project=results_dir, name=f'yolo11_{SET_ID}',",
    "            single_cls=True, verbose=True, device=train_device, batch=train_batch, workers=2, cache='ram')",
    "    print('=' * 100)"
]

# Apply with REAL newlines
nb["cells"][5]["source"] = [line + "\n" for line in c4_raw]
nb["cells"][6]["source"] = [line + "\n" for line in c5_raw]

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=4)
print("SUCCESS: Notebook fixed with real newlines.")
