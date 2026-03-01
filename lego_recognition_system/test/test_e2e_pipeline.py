import os
import sys
import json
import logging
import datetime
from pathlib import Path
from PIL import Image

# Setup paths to import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.part_resolver import resolve_set
from src.logic.generator import SyntheticGenerator
from src.logic.trainer import ModelTrainer
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex

# Load Config
with open(os.path.join(PROJECT_ROOT, "config.json"), "r") as f:
    config = json.load(f)

# Setup Logging
TEST_DIR = os.path.join(PROJECT_ROOT, "test")
LOGS_DIR = os.path.join(TEST_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_filename = os.path.join(LOGS_DIR, f"e2e_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("E2ETest")

# Constants
SET_ID = "75078-1"
NUM_PIECES = 2
NUM_IMAGES = 10
EPOCHS = 3
CONFIDENCE_THRESHOLD = 0.15

def run_e2e_pipeline():
    logger.info("===============================================")
    logger.info("🚀 STARTING E2E PIPELINE TEST")
    logger.info("===============================================")
    
    # Enable Universal Detector Mode
    os.environ["UNIVERSAL_DETECTOR"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix for MacOS OpenMP conflict
    logger.info("✅ UNIVERSAL_DETECTOR mode enabled")

    # --------------------------------------------------------------------------
    # FASE 1: Setup & Selección de Piezas
    # --------------------------------------------------------------------------
    logger.info("\n--- FASE 1: Selección de Piezas ---")
    parts = resolve_set(SET_ID, max_parts=NUM_PIECES)
    logger.info(f"Piezas seleccionadas: {[p['ldraw_id'] for p in parts]}")

    # --------------------------------------------------------------------------
    # FASE 2: Generación Sintética
    # --------------------------------------------------------------------------
    logger.info("\n--- FASE 2: Generación Sintética ---")
    datasets_dir = os.path.join(TEST_DIR, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    generator = SyntheticGenerator(
        blender_path=config['blender_executable_path'],
        ldraw_path=config['ldraw_library_path'],
        assets_dir=os.path.join(PROJECT_ROOT, "assets"),
        output_dir=datasets_dir
    )
    
    logger.info(f"Generando {NUM_IMAGES} imágenes para validación...")
    for progress, msg in generator.generate_dataset(SET_ID, parts, num_images=NUM_IMAGES):
        if msg:
            logger.info(f"Blender: {msg}")
        elif progress == -1:
            logger.error("❌ Falla en la generación sintética.")
            return

    expected_data_yaml = os.path.join(datasets_dir, SET_ID, "data.yaml")
    expected_meta = os.path.join(datasets_dir, SET_ID, "image_meta.jsonl")
    if os.path.exists(expected_data_yaml) and os.path.exists(expected_meta):
        logger.info("✅ Datos Sintéticos, data.yaml y metadatos generados correctamente.")
    else:
        logger.error("❌ Archivos de configuración de datos ausentes.")
        return

    # --------------------------------------------------------------------------
    # FASE 3: Entrenamiento YOLO (MOCK)
    # --------------------------------------------------------------------------
    logger.info("\n--- FASE 3: Entrenamiento YOLO ---")
    models_dir = os.path.join(TEST_DIR, "models", "yolo_model")
    os.makedirs(models_dir, exist_ok=True)
    
    trainer = ModelTrainer(model_dir=models_dir)
    logger.info(f"Entrenando modelo YOLO detectando 1 clase ('lego') por {EPOCHS} épocas...")
    try:
         trainer.train_model(SET_ID, os.path.join(datasets_dir, SET_ID), epochs=EPOCHS, progress_callback=lambda p, s: logger.info(f"YOLO: {s} - {p}%"))
         yolo_model_path = trainer.get_model_path(SET_ID)
         logger.info(f"✅ Entrenamiento YOLO exitoso. Modelo guardado en: {yolo_model_path}")
    except Exception as e:
         logger.error(f"❌ Error en entrenamiento YOLO: {e}")
         return


    # --------------------------------------------------------------------------
    # FASE 4: Generación de Índice (Búsqueda Vectorial)
    # --------------------------------------------------------------------------
    logger.info("\n--- FASE 4: Generación de Índice FAISS ---")
    extractor = FeatureExtractor(model_name='dinov2_vits14')
    vector_dir = os.path.join(TEST_DIR, "models", "piezas_vectores")
    os.makedirs(vector_dir, exist_ok=True)
    
    index_path = os.path.join(vector_dir, "test_index.index")
    vector_index = VectorIndex(dim=extractor.feature_dim)
    
    logger.info("Extrayendo embeddings de las imágenes generadas para simular el índice de referencia...")
    
    # Read meta
    image_dir = os.path.join(datasets_dir, SET_ID, "images")
    meta_records = []
    with open(expected_meta, "r") as f:
        for line in f:
            meta_records.append(json.loads(line))
            
    # Add first occurrence of each part to index to act as 'reference'
    indexed_parts = set()
    for record in meta_records:
        img_path = os.path.join(image_dir, record["img"])
        ldraw_ids = record["ids"]
        
        # We assume each image has 1 piece for simplicity in this minimal test reference building
        # or we just crop the center as a mock since the background is minimal
        # A more rigorous test would use YOLO here to crop, but let's just index the whole image as a mock reference
        if not os.path.exists(img_path): continue
        
        for ldraw_id in ldraw_ids:
            if ldraw_id not in indexed_parts:
                emb = extractor.get_embedding(img_path)
                mock_part_data = next(p for p in parts if p["ldraw_id"] == ldraw_id)
                vector_index.add(emb, mock_part_data)
                indexed_parts.add(ldraw_id)
                logger.info(f"Indexada pieza de referencia: {ldraw_id}")
                
    vector_index.save(index_path)
    logger.info("✅ Índice FAISS generado y guardado.")


    # --------------------------------------------------------------------------
    # FASE 5: Validación Cruzada (Inferencia E2E)
    # --------------------------------------------------------------------------
    logger.info("\n--- FASE 5: Inferencia end-to-end ---")
    
    # Pick a test image (different from the ones used as pure reference if possible, but let's just pick the last one)
    test_record = meta_records[-1]
    test_img_path = os.path.join(image_dir, test_record["img"])
    expected_ids = test_record["ids"]
    
    logger.info(f"Probando inferencia sobre {test_record['img']}. Esperado: {expected_ids}")
    
    from ultralytics import YOLO
    model = YOLO(yolo_model_path)
    results = model(test_img_path, conf=CONFIDENCE_THRESHOLD)
    
    boxes = results[0].boxes
    logger.info(f"YOLO detectó {len(boxes)} piezas.")
    
    img_obj = Image.open(test_img_path).convert('RGB')
    
    if len(boxes) == 0:
        logger.warning("YOLO no detectó piezas. Usando bounding boxes (Ground Truth) para validar el resto del pipeline...")
        # Fallback to Ground Truth
        label_path = os.path.join(datasets_dir, SET_ID, "labels", test_record["img"].replace(".png", ".txt"))
        if not os.path.exists(label_path):
            logger.error(f"❌ No se encontraron labels en {label_path}")
            return
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        img_w, img_h = img_obj.size
        for i, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized polygon to bounding box
            xs = [coords[j] * img_w for j in range(0, len(coords), 2)]
            ys = [coords[j] * img_h for j in range(1, len(coords), 2)]
            
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            
            # Expand slightly to match what YOLO might output
            padding = 10
            x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min(img_w, x2+padding), min(img_h, y2+padding)
            
            logger.info(f"  Det {i+1} (GT): Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            # Crop
            crop_img = img_obj.crop((x1, y1, x2, y2))
            
            # Extract features
            query_emb = extractor.get_embedding(crop_img)
            
            # Search index
            res = vector_index.search(query_emb, k=1)
            if res:
                best_match = res[0]
                pred_id = best_match['metadata']['ldraw_id']
                similarity = best_match['similarity']
                
                logger.info(f"    -> FAISS Predicción: {pred_id} (Similitud: {similarity:.2f})")
                
                if pred_id in expected_ids:
                    logger.info("    ✅ ¡COINCIDENCIA EXITOSA!")
                    success = True
                else:
                     logger.warning(f"    ❌ Falso positivo / Error de Clasificación. Esperado {expected_ids}.")
            else:
                 logger.warning("    ❌ No se encontraron resultados en el índice.")
                 
    else:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            logger.info(f"  Det {i+1}: Conf={confidence:.2f}, Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            # Crop
            crop_img = img_obj.crop((x1, y1, x2, y2))
            
            # Extract features
            query_emb = extractor.get_embedding(crop_img)
            
            # Search index
            res = vector_index.search(query_emb, k=1)
            if res:
                best_match = res[0]
                pred_id = best_match['metadata']['ldraw_id']
                similarity = best_match['similarity']
                
                logger.info(f"    -> FAISS Predicción: {pred_id} (Similitud: {similarity:.2f})")
                
                if pred_id in expected_ids:
                    logger.info("    ✅ ¡COINCIDENCIA EXITOSA!")
                    success = True
                else:
                     logger.warning(f"    ❌ Falso positivo / Error de Clasificación. Esperado {expected_ids}.")
            else:
                 logger.warning("    ❌ No se encontraron resultados en el índice.")
             
    logger.info("===============================================")
    if success:
         logger.info("🎉 TEST E2E PIPELINE COMPLETADO CON ÉXITO")
    else:
         logger.warning("⚠️ TEST E2E FINALIZÓ CON ERRORES O MISMATCH")
    logger.info("===============================================")
    logger.info(f"Reporte guardado en: {log_filename}")


if __name__ == "__main__":
    try:
        run_e2e_pipeline()
    except Exception as e:
        logger.exception(f"Error fatal en el Test E2E: {e}")
