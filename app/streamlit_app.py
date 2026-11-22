# ==============================================================================
# FRONTEND UTILISATEUR : STREAMLIT POUR DEEPBRAIN-DIAG
# Objectif : Permettre le téléversement d'une image IRM, appeler l'API FastAPI
#            (/predict/classify ou /predict/detect) et afficher les résultats.
# ==============================================================================

import io
import base64
import requests
from PIL import Image
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(
    page_title="DeepBrain-Diag | Diagnostic IRM",
    page_icon="🧠",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ==============================================================================
# EN-TÊTE ET BARRE LATÉRALE
# ==============================================================================
st.title("🧠 DeepBrain-Diag — Système d'Aide au Diagnostic Cérébral")
st.markdown(
    "Plateforme d'analyse d'images IRM combinant la **classification (GoogLeNet)** "
    "et la **détection/localisation d'objets (YOLOv8)**."
)

with st.sidebar:
    st.header("⚙️ Paramètres d'analyse")
    mode = st.radio(
        "Choisissez la fonctionnalité :",
        options=[
            "Classification (Type de tumeur)",
            "Détection & Localisation (Bounding Boxes)"
        ]
    )

    conf_threshold = 0.25
    if "Détection" in mode:
        conf_threshold = st.slider(
            "Seuil de confiance YOLOv8 :",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05
        )

    st.divider()
    # Vérification de l'état de l'API FastAPI
    try:
        health = requests.get(f"{API_URL}/", timeout=2)
        if health.status_code == 200:
            st.success("API FastAPI : En ligne")
        else:
            st.warning("API FastAPI : Réponse inattendue")
    except requests.exceptions.RequestException:
        st.error("API FastAPI : Hors ligne (lancez uvicorn)")

# ==============================================================================
# ZONE PRINCIPALE : TÉLÉVERSEMENT ET PRÉDICTION
# ==============================================================================
uploaded_file = st.file_uploader(
    "Téléversez une image IRM cérébrale (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image IRM chargée")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Résultat de l'analyse")

        if st.button("Lancer l'analyse", type="primary", use_container_width=True):
            with st.spinner("Analyse de l'image en cours via FastAPI..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                try:
                    # ----------------------------------------------------------
                    # MODE 1 : CLASSIFICATION (GoogLeNet)
                    # ----------------------------------------------------------
                    if "Classification" in mode:
                        response = requests.post(
                            f"{API_URL}/predict/classify",
                            files=files,
                            timeout=30
                        )
                        response.raise_for_status()
                        data = response.json()

                        st.metric(
                            label="Diagnostic prédit",
                            value=data["predicted_class"],
                            delta=f"Confiance : {data['confidence']} %"
                        )

                        st.markdown("**Distribution des probabilités par classe :**")
                        for cls_name, prob in data["probabilities"].items():
                            st.write(f"- **{cls_name}** : {prob} %")
                            st.progress(min(max(prob / 100.0, 0.0), 1.0))

                    # ----------------------------------------------------------
                    # MODE 2 : DÉTECTION ET LOCALISATION (YOLOv8)
                    # ----------------------------------------------------------
                    else:
                        response = requests.post(
                            f"{API_URL}/predict/detect",
                            files=files,
                            params={"conf_threshold": conf_threshold},
                            timeout=30
                        )
                        response.raise_for_status()
                        data = response.json()

                        # Décodage de l'image annotée retournée en Base64
                        img_bytes = base64.b64decode(data["annotated_image_base64"])
                        annotated_img = Image.open(io.BytesIO(img_bytes))
                        st.image(annotated_img, caption="Détection YOLOv8", use_container_width=True)

                        st.info(f"Nombre de lésions détectées : **{data['num_detections']}**")
                        for idx, det in enumerate(data["detections"], start=1):
                            st.write(
                                f"**#{idx} — {det['class']}** | Confiance : `{det['confidence']} %` "
                                f"| Coordonnées `[x1, y1, x2, y2]` : `{det['bbox_xyxy']}`"
                            )

                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de communication avec l'API FastAPI : {e}")