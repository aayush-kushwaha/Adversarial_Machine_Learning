import base64
import io

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from .attacks import fgsm_attack, pgd_attack
from .model import get_model
from .utils import tensor_to_pil, pil_to_base64


class PredictRequest(BaseModel):
    image_base64: str


class AttackRequest(BaseModel):
    image_base64: str
    attack_type: str
    epsilon: float
    alpha: float = 0.01
    iters: int = 10


def decode_image(image_data: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as exc:  # pragma: no cover - input validation
        raise HTTPException(status_code=400, detail="Invalid base64 string") from exc

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - input validation
        raise HTTPException(status_code=400, detail="Invalid image data") from exc


app = FastAPI(title="Adversarial CIFAR10 Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model_wrapper = get_model()
loss_fn = torch.nn.CrossEntropyLoss()


@app.post("/predict")
def predict(request: PredictRequest):
    image = decode_image(request.image_base64)
    tensor = model_wrapper.preprocess(image)
    predicted_class = model_wrapper.predict_class(tensor)
    return {"predicted_class": predicted_class}


@app.post("/attack")
def attack(request: AttackRequest):
    attack_type = request.attack_type.lower()
    image = decode_image(request.image_base64)
    tensor = model_wrapper.preprocess(image)
    original_class = model_wrapper.predict_class(tensor)
    labels = torch.tensor([original_class], device=model_wrapper.device)

    if attack_type == "fgsm":
        adversarial = fgsm_attack(
            model_wrapper.model,
            loss_fn,
            tensor,
            labels,
            epsilon=request.epsilon,
        )
    elif attack_type == "pgd":
        adversarial = pgd_attack(
            model_wrapper.model,
            loss_fn,
            tensor,
            labels,
            epsilon=request.epsilon,
            alpha=request.alpha,
            iters=request.iters,
        )
    else:
        raise HTTPException(status_code=400, detail="attack_type must be either FGSM or PGD")

    adversarial_class = model_wrapper.predict_class(adversarial)
    adv_image = tensor_to_pil(adversarial)
    adv_base64 = pil_to_base64(adv_image)

    return {
        "original_class": original_class,
        "adversarial_class": adversarial_class,
        "attack_type": attack_type.upper(),
        "eps": request.epsilon,
        "adversarial_image": adv_base64,
    }
