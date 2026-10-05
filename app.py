"""Streamlit inference and Grad-CAM UI for the proposed trained model."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from marine_plastic.explainability import grad_cam, overlay_grad_cam
from marine_plastic.models import create_model
from marine_plastic.preprocessing import transform_uploaded_image
from marine_plastic.training import choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
CHECKPOINT_CANDIDATES = (
    MODEL_DIR / "proposed_final.pt",
    MODEL_DIR / "selected_static.pt",
)


def available_checkpoint() -> Path | None:
    available = [path for path in CHECKPOINT_CANDIDATES if path.is_file()]
    return max(available, key=lambda path: path.stat().st_mtime_ns) if available else None


@st.cache_resource
def load_predictor(checkpoint_path: str) -> tuple[torch.nn.Module, list[str], torch.device]:
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["global_class_mapping"]
    model = create_model(
        len(class_names),
        pretrained=False,
        train_layer4=False,
        architecture=checkpoint["model_name"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names, device


st.set_page_config(page_title="Marine Debris Classifier", page_icon="🌊", layout="centered")
st.title("Marine Plastic Waste Classifier")
st.write("Upload an image to get the predicted label, confidence, and a Grad-CAM explanation.")
st.caption("This model supports multiple labels; the displayed prediction is the highest-scoring label.")

checkpoint_path = available_checkpoint()
if checkpoint_path is None:
    st.error(f"No trained checkpoint found under {MODEL_DIR}. Run `python train.py` first.")
else:
    uploaded = st.file_uploader("Choose a marine debris image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        model, classes, device = load_predictor(str(checkpoint_path))
        image_tensor = transform_uploaded_image(image, model.image_size).to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(image_tensor))[0]
        best_index = int(torch.argmax(probabilities).item())
        best_class = classes[best_index]
        confidence = float(probabilities[best_index].item())
        st.subheader(f"Predicted class: {best_class}")
        st.metric("Confidence", f"{confidence:.1%}")

        with st.expander("Scores for all labels"):
            for class_name, score in sorted(
                zip(classes, probabilities.tolist()), key=lambda item: item[1], reverse=True
            ):
                st.write(f"{class_name}: {score:.1%}")

        original, heatmap = grad_cam(model, image, best_index, device)
        st.subheader("Grad-CAM explanation")
        st.image(
            overlay_grad_cam(original, heatmap),
            caption="Highlighted regions contribute most to the selected class score.",
            use_container_width=True,
        )
