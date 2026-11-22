# 🧠 DeepBrain-Diag — Système Unifié d'Analyse et de Diagnostic des Tumeurs Cérébrales

![Python](https://img.shields.io/badge/Python-3.12%2F3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-GoogLeNet-orange?logo=tensorflow)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-purple)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?logo=mlflow)
![Airflow](https://img.shields.io/badge/Orchestration-Apache%20Airflow-017CEE?logo=apacheairflow)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)

**DeepBrain-Diag** est une solution complète de **Deep Learning** et **MLOps** conçue pour assister les spécialistes en imagerie médicale dans l'identification et la localisation automatisées des tumeurs cérébrales à partir d'images IRM.

Le système unifie deux approches complémentaires :
1. **Classification multi-classes** par Transfer Learning et Fine-Tuning avec **GoogLeNet (InceptionV1)** sous **TensorFlow/Keras** (`Glioma`, `Meningioma`, `No Tumor`, `Pituitary`).
2. **Détection et localisation spatiale** en temps réel par boîtes englobantes (*bounding boxes*) avec **YOLOv8n (Ultralytics)**.
3. **Architecture MLOps de bout en bout** intégrant **MLflow** (suivi des expériences et Model Registry), **Apache Airflow** (orchestration automatisée du pipeline), **FastAPI** (API REST) et **Streamlit** (interface interactive de diagnostic).

---

## 🏗️ Architecture du Projet

```text
DeepBrain-Diag/
├── api/
│   └── main.py                                       # Backend REST FastAPI (/predict/classify, /predict/detect)
├── app/
│   └── streamlit_app.py                              # Interface utilisateur interactive Streamlit
├── dags/
│   └── deepbrain_diag_pipeline.py                    # DAG Apache Airflow (6 tâches orchestrées + gestion d'erreurs)
├── reports/
│   ├── brain_tumor_yolov8/                           # Courbes d'évaluation, matrices de confusion et métriques YOLOv8
│   └── training_history_ft.csv                       # Historique d'entraînement GoogLeNet (Fine-Tuning)
├── data.yaml                                         # Configuration des ensembles Train/Val/Test et des classes YOLO
├── docker-compose.yaml                               # Déploiement local léger d'Apache Airflow (mode standalone)
├── neuro_diagnosis_classification_detection.ipynb    # Notebook complet : EDA, Prétraitement, Entraînement & MLflow
├── requirements.txt                                  # Dépendances Python du projet
├── README.md                                         # Documentation complète du projet
└── .gitignore                                        # Exclusion des datasets volumineux, poids et logs locaux
```

> **Note :** Les dossiers et fichiers locaux volumineux (`Data/`, `yolo_dataset/`, `models/`, `mlruns/`, `mlflow.db`, `logs/`) sont exclus du dépôt Git via `.gitignore`.

---

## 📊 Résultats et Performances des Modèles

Les performances ci-dessous sont enregistrées et vérifiées dans le registre **MLflow** du projet :

| Tâche / Modèle | Métriques Clés | Scores Obtenus |
| :--- | :--- | :--- |
| **Classification** (`GoogLeNet` Fine-Tuning) | Best Validation Accuracy / Train Accuracy | **96,58 %** / **98,23 %** |
| **Détection** (`YOLOv8n` — Jeu de Test) | mAP@0.5 / mAP@0.5:0.95 | **96,75 %** / **76,34 %** |
| **Détection** (`YOLOv8n` — Jeu de Test) | Precision / Recall | **93,80 %** / **96,80 %** |

---

## 🔬 Méthodologie Technique

### 1. Classification des Tumeurs Cérébrales (GoogLeNet — TensorFlow/Keras)
* **Exploration & Nettoyage :** Validation stricte des extensions d'images (`jpeg`, `jpg`, `bmp`, `png`) avec gestion des exceptions (`try-except`), exploration des 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`) et visualisation de la distribution des échantillons (*countplot*).
* **Partitionnement :** Découpage stratifié en **70 % Train**, **15 % Validation** et **15 % Test**.
* **Augmentation de données (Train uniquement) :** Application de retournements horizontaux (*flip*), flou gaussien (*blur*) et bruit aléatoire (*noise*) pour enrichir les échantillons et réduire le déséquilibre inter-classes.
* **Prétraitement & Chargement :** Redimensionnement en `224 × 224 × 3`, normalisation des pixels `[0, 1]` et chargement par lots via `tf.keras.utils.image_dataset_from_directory(..., shuffle=True)`.
* **Architecture & Entraînement :** Implémentation de **GoogLeNet (InceptionV1)** pré-entraîné avec adaptation de la tête de classification (`GlobalAveragePooling2D`, `Dropout`, `Dense` 4 classes `softmax`), entraînée en deux phases (**Feature Extraction** puis **Fine-Tuning**) avec l'optimiseur `Adam` et la perte `SparseCategoricalCrossentropy`.
* **Évaluation :** Suivi des courbes d'apprentissage, matrice de confusion, *Accuracy*, *Precision*, *Recall* et *F1-Score* (atteignant **96,58 %** de meilleure accuracy en validation).

### 2. Détection et Localisation des Tumeurs (YOLOv8n — Ultralytics)
* **Préparation & Intégrité des données :** Appariement rigoureux entre chaque image et son fichier d'annotation `.txt` (rejet des fichiers orphelins), vérification du format YOLO (`class_id x_center y_center width height`) et organisation dans `yolo_dataset/images/{train,val,test}` et `yolo_dataset/labels/{train,val,test}` pilotée par `data.yaml`.
* **Augmentation & Entraînement :** Configuration des augmentations géométriques et photométriques natives de YOLOv8 (`mosaic`, `fliplr`, `scale`, variations HSV) et entraînement de `yolov8n.pt`.
* **Évaluation & Généralisation :** Validation sur l'ensemble de test indépendant via les métriques **Precision** (`93,80 %`), **Recall** (`96,80 %`), **mAP@0.5** (`96,75 %`) et **mAP@0.5:0.95** (`76,34 %`).

### 3. Suivi des Expériences & Registre de Modèles (MLflow)
* Configuration d'un backend de suivi local (`sqlite:///mlflow.db`) et d'un répertoire d'artefacts (`./mlruns`).
* Enregistrement systématique des hyperparamètres (`learning_rate`, `epochs`, `batch_size`, `imgsz`, `optimizer`), des métriques d'entraînement/validation/test et des artefacts visuels (matrices de confusion, courbes F1/PR, fichiers `.csv`).
* Enregistrement des modèles retenus dans le **MLflow Model Registry** :
  * `DeepBrain_GoogLeNet_Classifier`
  * `DeepBrain_YOLOv8_Detector`

### 4. Orchestration du Pipeline (Apache Airflow)
Le DAG **`deepbrain_diag_training_pipeline`** (`dags/deepbrain_diag_pipeline.py`) automatise l'ensemble du flux en **6 tâches séquentielles** avec gestion des tentatives (`retries`), callback d'alerte (`on_failure_callback`) et journalisation détaillée :
1. `check_and_prepare_datasets` : Vérification et préparation des datasets et de `data.yaml`.
2. `validate_data_quality` : Validation de la structure et de la qualité des paires image/annotation YOLO.
3. `train_classification_model` : Entraînement / vérification du modèle de classification GoogLeNet.
4. `train_detection_model` : Entraînement / vérification du modèle de détection YOLOv8n.
5. `evaluate_and_log_mlflow` : Évaluation des modèles et enregistrement des résultats dans MLflow.
6. `save_models_and_artifacts` : Sauvegarde et validation finale des modèles et artefacts produits.

### 5. Déploiement API & Interface Utilisateur (FastAPI + Streamlit)
* **API REST FastAPI (`api/main.py`) :**
  * `GET /` : Vérification de l'état de santé du serveur et des modèles chargés.
  * `POST /predict/classify` : Prétraitement de l'image IRM et inférence GoogLeNet (retourne la classe prédite et la distribution des probabilités).
  * `POST /predict/detect` : Inférence YOLOv8 avec seuil de confiance configurable (retourne les coordonnées `[x1, y1, x2, y2]`, les scores et l'image annotée encodée en Base64).
* **Application Web Streamlit (`app/streamlit_app.py`) :**
  * Téléversement interactif d'images IRM (`JPG`, `JPEG`, `PNG`).
  * Bascule entre les modes **Classification** et **Détection & Localisation** avec réglage dynamique du seuil de confiance.
  * Communication HTTP avec FastAPI et affichage visuel des résultats.

---

## 🚀 Guide d'Installation et d'Exécution

### 1. Prérequis et Installation

```bash
git clone https://github.com/<votre-utilisateur>/DeepBrain-Diag.git
cd DeepBrain-Diag

# Installation des dépendances (Linux/Mac)
pip install -r requirements.txt

# Installation des dépendances (Windows PowerShell)
py -m pip install -r requirements.txt
```

### 2. Lancer l'interface MLflow (Suivi des expériences)

```bash
# Linux / Mac
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Windows PowerShell
py -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
* **Accès interface MLflow :** `http://127.0.0.1:5000`

### 3. Lancer l'orchestrateur Apache Airflow (via Docker)

```bash
# 1. Créer les répertoires nécessaires (Linux/Mac)
mkdir -p dags logs plugins

# 1. Créer les répertoires nécessaires (Windows PowerShell)
mkdir dags, logs, plugins

# 2. Démarrer les conteneurs Airflow en arrière-plan
docker compose up -d
```

* **Accès interface Airflow :** `http://localhost:8080`
* **DAG à activer :** `deepbrain_diag_training_pipeline`

**Récupérer le mot de passe administrateur Airflow :**
```powershell
# lire le fichier généré directement dans ce conteneur
docker exec deepbrain_airflow cat /opt/airflow/standalone_admin_password.txt

# Ou rechercher dans les logs de démarrage
docker compose logs | Select-String -Pattern "password"
```
*(Si votre configuration utilise les identifiants statiques par défaut, utilisez `airflow` / `airflow`.)*

**Arrêter proprement les conteneurs Airflow :**
```bash
docker compose down
```

### 4. Lancer l'API Backend FastAPI

```bash
# Linux / Mac
uvicorn api.main:app --reload --port 8000

# Windows PowerShell
py -m uvicorn api.main:app --reload --port 8000
```
* **Documentation interactive Swagger :** `http://127.0.0.1:8000/docs`

### 5. Lancer l'Interface Utilisateur Streamlit

Dans un second terminal (en gardant l'API FastAPI active) :

```bash
# Linux / Mac
streamlit run app/streamlit_app.py

# Windows PowerShell
py -m streamlit run app/streamlit_app.py
```
* **Accès application web :** `http://localhost:8501`

---

## 🛠️ Stack Technique

* **Deep Learning & Computer Vision :** TensorFlow / Keras (GoogLeNet), Ultralytics (YOLOv8), OpenCV, Pillow, NumPy, Pandas, Scikit-Learn
* **MLOps & Orchestration :** MLflow, Apache Airflow, Docker Compose
* **Backend & Frontend :** FastAPI, Uvicorn, Pydantic, Streamlit, Requests