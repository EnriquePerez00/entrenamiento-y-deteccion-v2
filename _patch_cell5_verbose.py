"""
One-shot patcher: inject verbose epoch callbacks into Cell 5 of patch_notebook.py.
Run: python3 _patch_cell5_verbose.py
"""

with open("patch_notebook.py", "r", encoding="utf-8") as f:
    src = f.read()

# ── Find the line that sets verbose=False ──
MARKER = "            single_cls=True, verbose=False,\\n\","

if MARKER not in src:
    print("❌ Marker not found. Current state may already be patched or changed.")
    print("Searching for 'verbose='...")
    for i, l in enumerate(src.split('\n')):
        if 'verbose' in l:
            print(f"  Line ~{i}: {l.strip()}")
    raise SystemExit(1)

# ── Callback lines to INSERT before the with timer.step block ──
CALLBACK_LINES = '''    \"    # ── Live epoch callback: prints progress to Kaggle output ──\\n\",
    \"    import time as _time\\n\",
    \"    _t = [_time.time()]\\n\",
    \"    def _on_start(trainer): _t[0] = _time.time()\\n\",
    \"    def _on_end(trainer):\\n\",
    \"        e, tot = trainer.epoch + 1, trainer.epochs\\n\",
    \"        elapsed = _time.time() - _t[0]\\n\",
    \"        m = trainer.metrics or {}\\n\",
    \"        box   = m.get(\'train/box_loss\', 0)\\n\",
    \"        cls_l = m.get(\'train/cls_loss\', 0)\\n\",
    \"        dfl   = m.get(\'train/dfl_loss\', 0)\\n\",
    \"        map50 = m.get(\'metrics/mAP50(B)\', 0)\\n\",
    \"        vram  = torch.cuda.memory_reserved(0)/1e9 if torch.cuda.is_available() else 0\\n\",
    \"        pct   = int(e/tot*20)\\n\",
    \"        bar   = chr(9608)*pct + chr(9617)*(20-pct)\\n\",
    \"        msg   = (f\'[{bar}] {e:>3}/{tot} | {elapsed:.1f}s | \'\\n\",
    \"                 f\'box={box:.4f} cls={cls_l:.4f} dfl={dfl:.4f} | \'\\n\",
    \"                 f\'mAP@50={map50:.4f} | VRAM={vram:.1f}GB\')\\n\",
    \"        print(msg, flush=True)\\n\",
    \"        logger.info(msg)\\n\",
    \"    model.add_callback(\'on_train_epoch_start\', _on_start)\\n\",
    \"    model.add_callback(\'on_train_epoch_end\', _on_end)\\n\",
    \"\\n\",
    \"    SEP = \'=\' * 90\\n\",
    \"    print(f\'\\\\\\\\n{SEP}\')\\n\",
    \"    print(f\'Entrenamiento: {num_gpus} GPU(s) | batch={train_batch*4} | cache=RAM | 50 epocas\')\\n\",
    \"    print(SEP)\\n\",
'''

# ── Find the insert position (just before the 'with timer.step' that wraps model.train) ──
TIMER_MARKER = "    \"    with timer.step('YOLO Training', metadata={'epochs': 50, 'gpus': num_gpus, 'batch': train_batch}):\\n\","

if TIMER_MARKER not in src:
    print("❌ timer.step marker not found.")
    raise SystemExit(1)

# Insert callbacks before the timer block
src = src.replace(TIMER_MARKER, CALLBACK_LINES + TIMER_MARKER)

# Change verbose=False to verbose=True and update timer metadata
src = src.replace(
    "            single_cls=True, verbose=False,\\n\",",
    "            single_cls=True, verbose=True,\\n\","
)
src = src.replace(
    "with timer.step('YOLO Training', metadata={'epochs': 50, 'gpus': num_gpus, 'batch': train_batch}):\\n\",",
    "with timer.step('YOLO Training', metadata={'epochs': 50, 'gpus': num_gpus, 'batch': train_batch*4}):\\n\","
)

with open("patch_notebook.py", "w", encoding="utf-8") as f:
    f.write(src)

print("✅ Verbose callbacks injected into Cell 5.")
