import base64
import io
import random

import requests
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

BACKEND_URL = "http://localhost:8000"

to_pil = transforms.ToPILImage()
transform = transforms.ToTensor()
testset = CIFAR10(root="./data", train=False, download=True, transform=transform)


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def sample_image(selected_class):
    while True:
        idx = random.randrange(len(testset))
        image, label = testset[idx]
        if selected_class == "Any" or label == selected_class:
            return image, label, idx


if "current_img" not in st.session_state:
    st.session_state["current_img"] = None
    st.session_state["current_label"] = None


st.title("Adversarial CIFAR-10 Demo")
st.write("Load CIFAR-10 test samples, predict them, and generate FGSM/PGD adversarial examples.")

class_filter = st.selectbox("Optional class filter", ["Any"] + list(range(10)))
load_button = st.button("Load Random CIFAR-10 Test Image")

if load_button:
    image_tensor, label, idx = sample_image(class_filter if class_filter == "Any" else int(class_filter))
    st.session_state["current_img"] = image_tensor
    st.session_state["current_label"] = label

if st.session_state["current_img"] is not None:
    original_pil = to_pil(st.session_state["current_img"])
    st.image(original_pil, caption=f"True label: {st.session_state['current_label']}", use_column_width=True)
else:
    st.info("Load a CIFAR-10 test image to begin.")

attack_type = st.selectbox("Attack", ["fgsm", "pgd"], index=0)
epsilon = st.slider("Epsilon", 0.0, 0.5, 0.1, 0.01)
alpha = None
iters = None
if attack_type == "pgd":
    alpha = st.slider("Alpha", 0.0, 0.5, 0.025, 0.005)
    iters = st.slider("Iterations", 1, 100, 20)

if st.button("Run Attack"):
    if st.session_state["current_img"] is None:
        st.warning("Load a CIFAR-10 image first.")
    else:
        pil_image = to_pil(st.session_state["current_img"])
        encoded = encode_image(pil_image)

        predict_resp = requests.post(
            f"{BACKEND_URL}/predict",
            json={"image_base64": encoded},
        )
        if not predict_resp.ok:
            st.error(predict_resp.text)
        else:
            predicted_class = predict_resp.json().get("predicted_class")
            st.write(f"Backend prediction: {predicted_class}")

            attack_payload = {
                "image_base64": encoded,
                "attack_type": attack_type,
                "epsilon": epsilon,
                "alpha": alpha if alpha is not None else 0.0,
                "iters": iters if iters is not None else 1,
            }
            attack_resp = requests.post(f"{BACKEND_URL}/attack", json=attack_payload)
            if attack_resp.ok:
                data = attack_resp.json()
                adv_image = decode_image(data["adversarial_image"])
                st.image(adv_image, caption=f"Adversarial image ({attack_type.upper()})", use_column_width=True)
                st.write(f"Original class: {data['original_class']}")
                st.write(f"Adversarial class: {data['adversarial_class']}")
            else:
                st.error(attack_resp.text)
