# Adversarial Project

This project wraps the CIFAR-10 model, preprocessing, FGSM, and PGD logic from adversarial_project.ipynb into a FastAPI backend with a Streamlit frontend so you can inspect predictions and adversarial attacks through HTTP endpoints or an interactive UI.

## Prerequisites

1. Place your trained weights at saved_model/model_cifar.pt. The backend builds SimpleCNN().to(device) in the same way as the notebook and immediately loads the state dict from this file.
2. Install dependencies with: pip install -r requirements.txt.

## Running the Backend

- cd adversarial_project
- uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

The backend exposes /predict and /attack endpoints. /attack returns the original prediction, adversarial prediction, epsilon, attack type, and a base64 PNG of the adversarial image. All tensor operations (model class, device selection, transforms, FGSM, PGD) are the exact code extracted from the notebook.

## Running the Frontend

- cd adversarial_project
- streamlit run frontend/app.py

The Streamlit UI lets you upload an image, request the backend prediction, choose FGSM or PGD, set epsilon (plus alpha/iterations for PGD), and visualize the adversarial output returned by the backend.

## Attack Overview

- FGSM perturbs the input once using adv = x + epsilon * sign(grad_x L(model(x), y)) and clamps the image to the [0,1] range.
- PGD applies multiple gradient steps of size alpha, projects the perturbation back to the L-infinity ball with radius epsilon, and clamps the pixels after each iteration.

Both implementations are copied verbatim from the notebook so numerical behavior matches your experiments.

## Notes

- The CIFAR-10 preprocessing uses the exact inference transform from the notebook: transforms.ToTensor() without extra normalization.
- The backend relies on model_cifar.pt being present; without it, initialization raises an error.
- Because the FastAPI service embeds the notebook code directly, any future changes to the notebook should be re-extracted to keep behavior aligned.
