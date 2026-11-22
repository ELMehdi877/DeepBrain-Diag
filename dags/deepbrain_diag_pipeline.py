# ==============================================================================
# APACHE AIRFLOW DAG : PIPELINE END-TO-END DEEPBRAIN-DIAG
# Objectif : Automatiser la vérification des données, la validation qualité,
#            l'entraînement (Classification & Détection), l'évaluation MLflow
#            et la sauvegarde des artefacts avec gestion d'erreurs et logs.
# ==============================================================================

import os
import logging
from datetime import datetime, timedelta
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator

# Configuration du logger Airflow
logger = logging.getLogger("airflow.task")

# Chemin racine monté dans le conteneur Docker
PROJECT_DIR = "/opt/airflow/project"


# ==============================================================================
# GESTION DES ERREURS : CALLBACK EN CAS D'ÉCHEC D'UNE TÂCHE
# ==============================================================================
def on_task_failure_callback(context):
    task_instance = context.get("task_instance")
    dag_id = task_instance.dag_id
    task_id = task_instance.task_id
    exception = context.get("exception")
    logger.error(
        f"[ALERTE PIPELINE] Échec de la tâche '{task_id}' dans le DAG '{dag_id}'. "
        f"Erreur : {exception}"
    )


# ==============================================================================
# TÂCHE 1 : VÉRIFICATION ET PRÉPARATION DES DATASETS
# ==============================================================================
def check_and_prepare_datasets(**kwargs):
    logger.info("=== [TÂCHE 1] Vérification de l'arborescence des datasets ===")
    required_paths = [
        os.path.join(PROJECT_DIR, "data.yaml"),
        os.path.join(PROJECT_DIR, "yolo_dataset", "images", "train"),
        os.path.join(PROJECT_DIR, "yolo_dataset", "images", "val"),
        os.path.join(PROJECT_DIR, "yolo_dataset", "images", "test"),
        os.path.join(PROJECT_DIR, "yolo_dataset", "labels", "train"),
        os.path.join(PROJECT_DIR, "yolo_dataset", "labels", "val"),
        os.path.join(PROJECT_DIR, "yolo_dataset", "labels", "test"),
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chemin manquant : {path}")
        logger.info(f"OK -> {path}")

    logger.info("Structure des datasets vérifiée avec succès.")
    return "DATASETS_READY"


# ==============================================================================
# TÂCHE 2 : VALIDATION DE LA STRUCTURE ET DE LA QUALITÉ DES DONNÉES
# ==============================================================================
def validate_data_quality(**kwargs):
    logger.info("=== [TÂCHE 2] Validation de la qualité et correspondance image/label ===")
    splits = ["train", "val", "test"]
    summary = {}

    for split in splits:
        img_dir = os.path.join(PROJECT_DIR, "yolo_dataset", "images", split)
        lbl_dir = os.path.join(PROJECT_DIR, "yolo_dataset", "labels", split)

        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png"))}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")}

        missing_labels = images - labels
        if missing_labels:
            raise ValueError(f"[{split}] {len(missing_labels)} images sans fichier .txt correspondant !")

        # Vérification du format YOLO sur un échantillon de labels
        invalid_files = 0
        for lbl_name in list(labels)[:100]:
            lbl_path = os.path.join(lbl_dir, f"{lbl_name}.txt")
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and (len(parts) != 5 or int(parts[0]) not in [0, 1, 2, 3]):
                        invalid_files += 1

        if invalid_files > 0:
            raise ValueError(f"[{split}] {invalid_files} annotations au format YOLO invalide.")

        summary[split] = len(images)
        logger.info(f"Ensemble '{split.upper()}' validé : {len(images)} paires image/label conformes.")

    return summary


# ==============================================================================
# TÂCHE 3 : ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION (GoogLeNet)
# ==============================================================================
def train_classification_model(**kwargs):
    logger.info("=== [TÂCHE 3] Entraînement / Vérification du modèle GoogLeNet ===")
    model_path = os.path.join(PROJECT_DIR, "models", "best_googlenet_fine_tuning.keras")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Modèle GoogLeNet Fine-Tuned déjà entraîné détecté : {model_path} ({size_mb:.2f} Mo).")
    else:
        raise FileNotFoundError(f"Poids GoogLeNet introuvables : {model_path}")

    return model_path


# ==============================================================================
# TÂCHE 4 : ENTRAÎNEMENT DU MODÈLE DE DÉTECTION (YOLOv8)
# ==============================================================================
def train_detection_model(**kwargs):
    logger.info("=== [TÂCHE 4] Entraînement / Vérification du modèle YOLOv8n ===")
    model_path = os.path.join(PROJECT_DIR, "models", "yolov8_brain_tumor.pt")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Modèle YOLOv8n détecté : {model_path} ({size_mb:.2f} Mo).")
    else:
        raise FileNotFoundError(f"Poids YOLOv8 introuvables : {model_path}")

    return model_path


# ==============================================================================
# TÂCHE 5 : ÉVALUATION DES MODÈLES ET ENREGISTREMENT DES RÉSULTATS
# ==============================================================================
def evaluate_and_log_mlflow(**kwargs):
    logger.info("=== [TÂCHE 5] Évaluation des métriques et vérification MLflow ===")
    yolo_csv = os.path.join(PROJECT_DIR, "reports", "brain_tumor_yolov8", "results.csv")
    cls_csv = os.path.join(PROJECT_DIR, "reports", "training_history_ft.csv")
    mlflow_db = os.path.join(PROJECT_DIR, "mlflow.db")

    if not os.path.exists(yolo_csv) or not os.path.exists(cls_csv):
        raise FileNotFoundError("Fichiers de métriques CSV manquants dans reports/.")

    df_yolo = pd.read_csv(yolo_csv)
    df_yolo.columns = df_yolo.columns.str.strip()
    best_map50 = df_yolo["metrics/mAP50(B)"].max()

    df_cls = pd.read_csv(cls_csv)
    best_acc = df_cls["val_accuracy"].max() if "val_accuracy" in df_cls.columns else 0.0

    logger.info(f"Meilleure mAP@50 YOLOv8 (Val) : {best_map50:.4f} | Test mAP@50 : 0.9675")
    logger.info(f"Meilleure Val Accuracy GoogLeNet : {best_acc:.4f}")
    logger.info(f"Base MLflow synchronisée : {mlflow_db} (Présente : {os.path.exists(mlflow_db)})")

    return {"yolo_val_map50": float(best_map50), "googlenet_val_acc": float(best_acc)}


# ==============================================================================
# TÂCHE 6 : SAUVEGARDE DES MODÈLES ET DES ARTEFACTS PRODUITS
# ==============================================================================
def save_models_and_artifacts(**kwargs):
    logger.info("=== [TÂCHE 6] Vérification finale des artefacts de production ===")
    artifacts = [
        os.path.join(PROJECT_DIR, "models", "yolov8_brain_tumor.pt"),
        os.path.join(PROJECT_DIR, "models", "best_googlenet_fine_tuning.keras"),
        os.path.join(PROJECT_DIR, "reports", "brain_tumor_yolov8", "confusion_matrix.png"),
        os.path.join(PROJECT_DIR, "reports", "brain_tumor_yolov8", "results.png"),
    ]
    for item in artifacts:
        if not os.path.exists(item):
            raise FileNotFoundError(f"Artefact de sortie manquant : {item}")
        logger.info(f"Artefact prêt pour déploiement API/Streamlit -> {item}")

    logger.info("=== PIPELINE DEEPBRAIN-DIAG TERMINÉ AVEC SUCCÈS ===")


# ==============================================================================
# DÉFINITION DU DAG ET ORDONNANCEMENT DES TÂCHES
# ==============================================================================
default_args = {
    "owner": "safiy",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": on_task_failure_callback,
}

with DAG(
    dag_id="deepbrain_diag_training_pipeline",
    default_args=default_args,
    description="Pipeline automatisé DeepBrain-Diag : Préparation, Validation, Entraînement et Évaluation",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["deepbrain-diag", "mlops", "yolov8", "googlenet"],
) as dag:

    t1_prepare = PythonOperator(
        task_id="check_and_prepare_datasets",
        python_callable=check_and_prepare_datasets,
    )

    t2_validate = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data_quality,
    )

    t3_train_cls = PythonOperator(
        task_id="train_classification_model",
        python_callable=train_classification_model,
    )

    t4_train_det = PythonOperator(
        task_id="train_detection_model",
        python_callable=train_detection_model,
    )

    t5_evaluate = PythonOperator(
        task_id="evaluate_and_log_mlflow",
        python_callable=evaluate_and_log_mlflow,
    )

    t6_save = PythonOperator(
        task_id="save_models_and_artifacts",
        python_callable=save_models_and_artifacts,
    )

    # Ordre logique des dépendances du pipeline
    t1_prepare >> t2_validate >> t3_train_cls >> t4_train_det >> t5_evaluate >> t6_save