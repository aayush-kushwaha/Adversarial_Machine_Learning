
# 📘 Adversarial Attacks on CIFAR-10  [Code Improvements & Fix Summary]

This document explains **all modifications** made to the original FGSM + PGD adversarial attack notebook.
The goal is to provide a clear, technical, and developer-friendly summary suitable for team members, instructors, and documentation.

---

## 🔧 Overview

The original notebook implemented:

* CIFAR-10 loading
* A simple CNN training loop
* FGSM and PGD adversarial attacks
* Evaluation of accuracy & attack success

While functional, it had multiple limitations and runtime issues.
This README outlines **what was changed**, **why it was necessary**, and **how the fixes improve reliability and analysis quality**.

---

## 🚀 Key Improvements

### 1. **Multi-Epsilon FGSM & PGD Attacks**

**Original behavior:**

```python
epsilon = 0.03
```

Only a single attack strength was supported.

**Issue:**
A proper adversarial robustness analysis requires **multiple epsilons** to evaluate how accuracy changes as perturbations increase.

**Fix:**

```python
epsilons = [0.03, 0.1, 0.2]
```

FGSM and PGD attacks now run for *all* epsilon values, and results are stored for comparison.

**Benefit:**

* Enables weak, medium, and strong attack evaluation
* Produces more meaningful ASR curves
* Matches academic/industry evaluation standards

---

### 2. **Robust Visualization System**

The original visualization crashed due to:

* using `.numpy()` on tensors that required gradients
* incorrect tensor shapes
* trying to index more images than provided

**Fixes added:**

#### ✔ Automatic `.detach()` and `.cpu()`

Guarantees tensors are safe for plotting:

```python
tensor.detach().cpu().permute(1,2,0)
```

#### ✔ Perturbation Heatmap

Shows magnitude of adversarial noise:

```python
diff = (adversarial - original).abs() * scale
```

#### ✔ 10-Image Side-by-Side Grid

Displays multiple original vs adversarial samples.

**Benefits:**

* No more runtime errors
* High-quality visual explanations for reports
* Enables clearer understanding of model vulnerabilities

---

### 3. **Device-Safe Evaluation (Fix for GPU/CPU Mismatch)**

**Original error:**

```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
```

**Cause:**

* Model was on **GPU**
* Adversarial images were accidentally kept on **CPU**
* PyTorch never auto-moves tensors → mismatch

**Fix:**
Inside `evaluate_attack()`:

```python
adv_batch = adv_images[j:j+len(images)].to(device)
adv_pred = model(adv_batch).argmax(1)
```

**Benefit:**

* Works on **GPU** and **CPU** automatically
* Eliminates device mismatch crashes
* Makes evaluation fully stable

---

### 4. **Structured & Clean Attack Pipeline**

The notebook was reorganized for clarity:

* Attack generation separated from visualization
* FGSM/PGD results stored in structured dictionaries:

  ```python
  fgsm_results[eps] = adv_images
  ```
* Visualization placed at the end
* Added “experiment-ready” output blocks

**Benefit:**
The notebook now resembles a proper adversarial machine learning experiment paper:

* Training
* Multi-epsilon FGSM
* Multi-epsilon PGD
* Evaluation
* Visualization
* Heatmaps
* Comparison grid

---

## 🛠 Summary of All Fixes

| Category              | Issue in Original             | Fix Applied                           | Result                          |
| --------------------- | ----------------------------- | ------------------------------------- | ------------------------------- |
| **FGSM/PGD attacks**  | Only one epsilon              | Added multi-epsilon loops             | Full robustness analysis        |
| **Visualization**     | Shape errors, grad errors     | Added `.detach()`, heatmap, grid view | Reliable and clear plots        |
| **Device handling**   | GPU/CPU mismatch              | Added `.to(device)` for adv images    | Device-safe evaluation          |
| **Code organization** | Mixed logic, limited analysis | Structured & modular pipeline         | Clean, reproducible notebook    |
| **Interpretability**  | Noise not visible             | Added perturbation heatmap            | Better understanding of attacks |

---

## 📈 Final Result

After applying all improvements, the notebook:

* Trains a CNN
* Generates FGSM & PGD attacks at multiple strengths
* Evaluates Clean Accuracy, Adversarial Accuracy, ASR
* Visualizes:

  * original vs adversarial images
  * perturbation heatmaps
  * 10-image attack comparison grid
* Works on GPU and CPU without modification
* Produces research-quality analysis

---
