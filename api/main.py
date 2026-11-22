# ==============================================================================
# BACKEND API : FASTAPI POUR DEEPBRAIN-DIAG
# Objectif : Exposer deux endpoints HTTP (/predict/classify et /predict/detect)
#            pour servir les modèles GoogLeNet et YOLOv8 entraînés.
# ==============================================================================

import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException

# 1. Initialisation de l'application FastAPI
app = FastAPI(
    title="DeepBrain-Diag API",
    description="API de classification (GoogLeNet) et détection (YOLOv8) des tumeurs cérébrales.",
    version="1.0.0"
)

# 2. Chargement des modèles au démarrage du serveur
CLS_MODEL_PATH = "./models/best_googlenet_fine_tuning.keras"
DET_MODEL_PATH = "./models/yolov8_brain_tumor.pt"

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

print("Chargement des modèles DeepBrain-Diag...")
cls_model = tf.keras.models.load_model(CLS_MODEL_PATH)
det_model = YOLO(DET_MODEL_PATH)
print("Modèles chargés avec succès !")


# ==============================================================================
# ENDPOINT 0 : VÉRIFICATION DE L'ÉTAT DU SERVEUR (HEALTHCHECK)
# ==============================================================================
@app.get("/")
def health_check():
    return {"status": "online", "models": ["GoogLeNet", "YOLOv8n"]}


# ==============================================================================
# ENDPOINT 1 : CLASSIFICATION DE L'IMAGE CÉRÉBRALE (GoogLeNet)
# ==============================================================================
@app.post("/predict/classify")
async def predict_classification(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Détection automatique de la taille d'entrée attendue par le modèle Keras
        input_shape = cls_model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        target_h, target_w = input_shape[1] or 224, input_shape[2] or 224

        image_resized = image.resize((target_w, target_h))
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Prédiction
        preds = cls_model.predict(img_batch, verbose=0)
        if isinstance(preds, dict):
            preds = list(preds.values())[0]
        elif isinstance(preds, list):
            preds = preds[0]

        probs = np.array(preds)[0].tolist()
        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la classification : {str(e)}")


# ==============================================================================
# ENDPOINT 2 : DÉTECTION ET LOCALISATION DES TUMEURS (YOLOv8)
# ==============================================================================
@app.post("/predict/detect")
async def predict_detection(file: UploadFile = File(...), conf_threshold: float = 0.25):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Inférence YOLOv8
        results = det_model.predict(source=image, conf=conf_threshold, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])
            coords = [round(float(x), 2) for x in box.xyxy[0].tolist()]

            detections.append({
                "class": label,
                "confidence": round(conf * 100, 2),
                "bbox_xyxy": coords
            })

        # Génération de l'image avec boîtes englobantes (BGR -> RGB -> Base64)
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image_base64": img_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection : {str(e)}")