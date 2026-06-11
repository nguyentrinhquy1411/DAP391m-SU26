# Evidential Deep Learning (EDL) Classifier: Inputs and Outputs

This document describes the exact mathematical and feature-engineering representation of the inputs and outputs for the **Evidential Deep Learning (EDL) Posture Classifier** used in the AES-RARR framework. This content is structured for direct inclusion in the methodology and implementation sections of the academic paper.

---

## 1. Feature Engineering Inputs
For each target $i$ detected at time $t$ in a fused RGB-LWIR frame, the classifier receives a **7-dimensional geometric feature vector** $\mathbf{x}_{i,t} \in \mathbb{R}^7$. These features are engineered to remain invariant to raw image textures while capturing scale and location signals:

| Feature Dimension | Symbol | Description | Mathematical Definition |
| :--- | :---: | :--- | :--- |
| 1. Normalized Width | $w_{\text{norm}}$ | Bounding box width scaled by frame width | $w_{\text{bbox}} / W_{\text{frame}}$ |
| 2. Normalized Height | $h_{\text{norm}}$ | Bounding box height scaled by frame height | $h_{\text{bbox}} / H_{\text{frame}}$ |
| 3. Aspect Ratio | $r_{\text{asp}}$ | Ratio of crop width to height | $w_{\text{bbox}} / (h_{\text{bbox}} + 10^{-6})$ |
| 4. Normalized Center X | $cx_{\text{norm}}$ | Centroid X-coordinate normalized | $(x_{\text{bbox}} + w_{\text{bbox}}/2.0) / W_{\text{frame}}$ |
| 5. Normalized Center Y | $cy_{\text{norm}}$ | Centroid Y-coordinate normalized | $(y_{\text{bbox}} + h_{\text{bbox}}/2.0) / H_{\text{frame}}$ |
| 6. Center Distance | $d_{\text{center}}$ | Euclidean distance from image principal point | $\sqrt{(cx_{\text{norm}} - 0.5)^2 + (cy_{\text{norm}} - 0.5)^2}$ |
| 7. Relative Crop Area | $a_{\text{rel}}$ | Crop area divided by total frame area | $(w_{\text{bbox}} \times h_{\text{bbox}}) / (W_{\text{frame}} \times H_{\text{frame}})$ |

---

## 2. Evidential Classifier Outputs
Unlike deterministic classifiers that output class probabilities via a Softmax head, the EDL posture classifier employs a **Softplus activation** on the final linear layer to produce non-negative evidence parameters. These parameterise a Dirichlet distribution, enabling single-pass quantification of epistemic uncertainty.

### 2.1 Evidence and Dirichlet Parameters
* **Evidence Vector ($\mathbf{e}_{i,t} \in \mathbb{R}^K_{\geq 0}$)**: The raw positive output from the Softplus activation for $K$ classes:
  $$\mathbf{e}_{i,t} = \operatorname{Softplus}(\mathbf{z}_{i,t}) = \ln(1 + \exp(\mathbf{z}_{i,t}))$$
* **Dirichlet Parameters ($\boldsymbol{\alpha}_{i,t} \in \mathbb{R}^K_{> 1}$)**:
  $$\alpha_{i,t,k} = e_{i,t,k} + 1$$
* **Total Dirichlet Strength ($S_{i,t} \in \mathbb{R}$)**:
  $$S_{i,t} = \sum_{k=1}^K \alpha_{i,t,k} = \sum_{k=1}^K e_{i,t,k} + K$$

### 2.2 Subjective Logic Decomposition
Using the Subjective Logic framework, the total evidence is decomposed into belief masses and epistemic uncertainty:
* **Belief Masses ($b_{i,t,k} \in [0, 1]$)**: The portion of belief assigned specifically to class $k$:
  $$b_{i,t,k} = \frac{e_{i,t,k}}{S_{i,t}}$$
* **Epistemic Uncertainty ($u_{i,t} \in [0, 1]$)**: The vacant belief representing overall ignorance or lack of evidence (e.g., due to wave occlusion or excessive distance):
  $$u_{i,t} = \frac{K}{S_{i,t}}$$
  *(Note that $\sum_{k=1}^K b_{i,t,k} + u_{i,t} = 1$ is always mathematically satisfied).*
* **Expected Probabilities ($\hat{p}_{i,t,k} \in [0, 1]$)**: The expected value of the probability simplex under the Dirichlet distribution:
  $$\hat{p}_{i,t,k} = \frac{\alpha_{i,t,k}}{S_{i,t}}$$

---

## 3. Classification Categories ($K$)

The framework is evaluated across two distinct category configurations depending on the track:

### 3.1 Posture Distress States ( mSAR RARR Study, $K=4$ )
Used in the risk-averse rescue routing simulation:
1. **Drowning** (Class 0, Risk Weight = 1.0, high decay rate)
2. **Floating** (Class 1, Risk Weight = 0.4)
3. **Swimming** (Class 2, Risk Weight = 0.2)
4. **PFD Floater** (Class 3, Risk Weight = 0.1, low decay rate)

### 3.2 Real SeaDronesSee Categories ( Object Detection Track, $K=5$ )
Parsed directly from the COCO annotations in `instances_val.json`:
1. `swimmer` (Class 0)
2. `boat` (Class 1)
3. `buoy` (Class 2)
4. `life_saving_appliances` (Class 3)
5. `jetski` (Class 4)
