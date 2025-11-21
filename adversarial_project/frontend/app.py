import base64
import io
import random

import requests
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

BACKEND_URL = "http://localhost:8000"
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

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


def badge(text, color):
    return f"<span style='color:{color};font-weight:bold'>{text}</span>"


def card_start(title):
    return (
        "<div style='background:#111;padding:15px 18px;border-radius:14px;border:1px solid #333;"
        "box-shadow:0 0 12px rgba(0,255,255,0.15);margin-bottom:15px;'>"
        f"<h4 style='margin-bottom:8px;color:#67e8f9;'>{title}</h4>"
    )


def card_end():
    return "</div>"


if "current_img" not in st.session_state:
    st.session_state["current_img"] = None
    st.session_state["true_label"] = None
    st.session_state["clean_pred"] = None
    st.session_state["adv_pred"] = None
    st.session_state["adv_image"] = None


st.set_page_config(page_title="Dark Neon Attack Lab", layout="wide")
st.title("⚡ Dark Neon Attack Lab")
st.write("Interactively run FGSM and PGD adversarial attacks on CIFAR-10.")

controls = st.container()
with controls:
    st.markdown(
        "<div style='background:#0b0b0b;border:1px solid #1f2937;padding:20px;border-radius:16px;"
        "box-shadow:0 0 15px rgba(0,255,255,0.08);'>",
        unsafe_allow_html=True,
    )
    class_filter = st.selectbox("Class filter", ["Any"] + list(range(10)))
    load_button = st.button("🔁 Load Random CIFAR-10 Test Image")
    attack_type = st.selectbox("Attack", ["fgsm", "pgd"], index=0)
    epsilon = st.selectbox("Epsilon (Attack Strength)", [0.03, 0.1, 0.2], index=0)
    alpha = epsilon / 4 if attack_type == "pgd" else None
    if attack_type == "pgd":
        st.markdown(f"Alpha (ε/4): **{alpha:.4f}**")
        iters = st.selectbox("Iterations", [10, 20, 40], index=1)
    else:
        iters = 1
    st.markdown(card_end(), unsafe_allow_html=True)

if load_button:
    selected = class_filter if class_filter == "Any" else int(class_filter)
    image_tensor, label, _ = sample_image(selected)
    st.session_state["current_img"] = image_tensor
    st.session_state["true_label"] = label
    st.session_state["clean_pred"] = None
    st.session_state["adv_pred"] = None
    st.session_state["adv_image"] = None

if st.button("🚀 Run Attack"):
    if st.session_state["current_img"] is None:
        st.warning("Load a CIFAR-10 image first.")
    else:
        pil_image = to_pil(st.session_state["current_img"])
        encoded = encode_image(pil_image)

        predict_resp = requests.post(f"{BACKEND_URL}/predict", json={"image_base64": encoded})
        if not predict_resp.ok:
            st.error(predict_resp.text)
        else:
            clean_pred = predict_resp.json().get("predicted_class")
            st.session_state["clean_pred"] = clean_pred

            attack_payload = {
                "image_base64": encoded,
                "attack_type": attack_type,
                "epsilon": epsilon,
                "alpha": alpha if alpha is not None else 0.0,
                "iters": iters,
            }
            attack_resp = requests.post(f"{BACKEND_URL}/attack", json=attack_payload)
            if attack_resp.ok:
                data = attack_resp.json()
                st.session_state["adv_pred"] = data.get("adversarial_class")
                st.session_state["adv_image"] = decode_image(data["adversarial_image"])
            else:
                st.error(attack_resp.text)

col1, col2 = st.columns(2)

if st.session_state["current_img"] is not None:
    true_label = st.session_state["true_label"]
    true_name = classes[true_label]
    clean_pred = st.session_state["clean_pred"]
    clean_name = classes[clean_pred] if clean_pred is not None else "?"
    clean_match = clean_pred is not None and clean_pred == true_label

    with col1:
        st.markdown(card_start("Clean Sample"), unsafe_allow_html=True)
        st.image(to_pil(st.session_state["current_img"]), use_container_width=True)
        clean_color = "#00ff99" if clean_match else "#ff4444"
        st.markdown(
            f"True: {badge(true_name, '#00ff99')}<br>"
            f"Predicted: {badge(clean_name, clean_color)} {'✓' if clean_match else '✕'}",
            unsafe_allow_html=True,
        )
        st.markdown(card_end(), unsafe_allow_html=True)
else:
    st.info("Load a CIFAR-10 test image to visualize results.")

if st.session_state["adv_image"] is not None:
    adv_pred = st.session_state["adv_pred"]
    adv_name = classes[adv_pred] if adv_pred is not None else "?"
    clean_pred = st.session_state["clean_pred"]
    clean_name = classes[clean_pred] if clean_pred is not None else "?"
    fooled = adv_pred is not None and clean_pred is not None and adv_pred != clean_pred

    with col2:
        st.markdown(card_start("Adversarial Sample"), unsafe_allow_html=True)
        st.image(st.session_state["adv_image"], use_container_width=True)
        fool_color = "#ff4444" if fooled else "#ffaa00"
        st.markdown(
            f"After Attack: {badge(adv_name, '#ff4444')}<br>"
            f"Fooled: {badge('YES (✓)', fool_color) if fooled else badge('NO (✕)', fool_color)}",
            unsafe_allow_html=True,
        )
        st.markdown(card_end(), unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#9ca3af;margin-top:30px;'>"
    "Adversarial ML Demo — FGSM & PGD on CIFAR-10"
    "</p>",
    unsafe_allow_html=True,
)
