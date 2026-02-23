import time
import logging
import json
from contextlib import contextmanager

logger = logging.getLogger("LegoVision")

class PipelineTimer:
    """Utility to measure and log execution time of pipeline steps."""
    
    _start_times = {}
    _performance_data = {
        "hardware": {},
        "steps": [],
        "configurations": {}
    }

    @staticmethod
    def detect_hardware():
        """Detect and log current hardware specifications."""
        import torch, platform, psutil
        hw = {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "os": platform.system(),
            "gpu_available": torch.cuda.is_available(),
            "gpus": []
        }
        if hw["gpu_available"]:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                hw["gpus"].append({
                    "name": p.name,
                    "vram_gb": round(p.total_memory / (1024**3), 2)
                })
        PipelineTimer._performance_data["hardware"] = hw
        logger.info(f"💻 Hardware detectado: {hw}")
        PipelineTimer._flush_logs()

    @staticmethod
    @contextmanager
    def step(step_name, metadata=None):
        """Context manager to time a step and log its results."""
        logger.info(f"▶️ INICIANDO PASO: {step_name}")
        PipelineTimer._flush_logs()
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ PASO COMPLETADO: {step_name} | Duración: {duration:.2f}s")
            PipelineTimer._performance_data["steps"].append({
                "name": step_name,
                "duration_sec": round(duration, 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": metadata or {}
            })
            PipelineTimer._flush_logs()
            
    @staticmethod
    def log_config(config_dict):
        """Store the pipeline configuration used."""
        PipelineTimer._performance_data["configurations"].update(config_dict)

    @staticmethod
    def save_report(output_path):
        """Saves a structured JSON report of the pipeline performance."""
        with open(output_path, 'w') as f:
            json.dump(PipelineTimer._performance_data, f, indent=4)
        logger.info(f"📊 Informe de rendimiento guardado en: {output_path}")

    @staticmethod
    def start(step_name):
        PipelineTimer._start_times[step_name] = time.time()
        logger.info(f"▶️ INICIANDO: {step_name}")
        PipelineTimer._flush_logs()

    @staticmethod
    def stop(step_name):
        if step_name in PipelineTimer._start_times:
            duration = time.time() - PipelineTimer._start_times[step_name]
            logger.info(f"✅ COMPLETADO: {step_name} | Duración: {duration:.2f}s")
            del PipelineTimer._start_times[step_name]
            PipelineTimer._flush_logs()
        else:
            logger.warning(f"⚠️ Intentando detener un paso no iniciado: {step_name}")

    @staticmethod
    def _flush_logs():
        """Force all logging handlers to write to disk."""
        for handler in logger.handlers:
            handler.flush()
        # Also flush standard streams just in case
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
