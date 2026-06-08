import json

# Define the cells for the Jupyter Notebook
cells = []

# Cell 1: Title and Intro (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Active Evidential Sensing and Risk-Averse Rescue Routing (AES-RARR)\n",
        "### Maritime Search and Rescue AI Research Assistant\n",
        "\n",
        "This notebook provides a complete functional demonstration of the **AES-RARR** framework:\n",
        "1. **Exploratory Data Analysis (EDA):** Analyzes the SeaDronesSee annotation distributions and UAV telemetry.\n",
        "2. **Evidential Deep Learning (EDL) Pipeline:** Implements a Dirichlet-based classifier that outputs posture belief masses alongside epistemic uncertainty, including Expected Calibration Error (ECE) monitoring.\n",
        "3. **Closed-Loop Tracker & Prioritization Simulator:** Updates target state parameters under ocean current drift using a dynamic, uncertainty-scaled Kalman Filter, and routes rescue resources via a Risk-UCB priority engine."
    ]
})

# Cell 2: Setup (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install kagglehub if needed (for downloading SeaDronesSee)\n",
        "try:\n",
        "    import kagglehub\n",
        "    print(\"kagglehub is already installed.\")\n",
        "except ImportError:\n",
        "    print(\"Installing kagglehub...\")\n",
        "    !pip install -q kagglehub\n",
        "\n",
        "import os\n",
        "import json\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt"
    ]
})

# Cell 3: EDA Section Intro (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Part 1: Exploratory Data Analysis (EDA)\n",
        "We parse the SeaDronesSee COCO JSON annotations. If the dataset isn't downloaded, we generate a mock dataset to allow dry-run execution."
    ]
})

# Cell 4: EDA Logic (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "DATASET_JSON = \"/content/sds-dataset/annotations/instances_val.json\"\n",
        "\n",
        "def generate_mock_coco_json(output_path):\n",
        "    categories = [\n",
        "        {\"id\": 1, \"name\": \"swimmer\"},\n",
        "        {\"id\": 2, \"name\": \"floater\"},\n",
        "        {\"id\": 3, \"name\": \"boat\"},\n",
        "        {\"id\": 4, \"name\": \"life jacket\"},\n",
        "        {\"id\": 5, \"name\": \"buoy\"}\n",
        "    ]\n",
        "    images = []\n",
        "    annotations = []\n",
        "    np.random.seed(42)\n",
        "    for img_id in range(1, 101):\n",
        "        alt = float(np.random.choice([10, 20, 30, 50, 80, 120, 180, 240]))\n",
        "        pitch = float(np.random.uniform(20, 90))\n",
        "        images.append({\n",
        "            \"id\": img_id, \"width\": 1920, \"height\": 1080,\n",
        "            \"file_name\": f\"val/{img_id:04d}.jpg\",\n",
        "            \"altitude\": alt, \"gimbal_pitch\": pitch,\n",
        "            \"latitude\": 20.84, \"longitude\": 107.03\n",
        "        })\n",
        "        num_targets = np.random.randint(1, 5)\n",
        "        for _ in range(num_targets):\n",
        "            cat_id = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])\n",
        "            w = float(np.random.exponential(scale=30.0)) + 5.0\n",
        "            h = w * np.random.uniform(0.8, 1.2)\n",
        "            annotations.append({\n",
        "                \"id\": len(annotations) + 1, \"image_id\": img_id,\n",
        "                \"category_id\": int(cat_id), \"bbox\": [100, 100, w, h],\n",
        "                \"area\": float(w * h)\n",
        "            })\n",
        "    coco_mock = {\"images\": images, \"annotations\": annotations, \"categories\": categories}\n",
        "    os.makedirs(os.path.dirname(output_path), exist_ok=True)\n",
        "    with open(output_path, 'w') as f:\n",
        "        json.dump(coco_mock, f, indent=2)\n",
        "    print(f\"Created mock dataset at {output_path}\")\n",
        "\n",
        "# Trigger mock generation if real dataset is not yet downloaded\n",
        "if not os.path.exists(DATASET_JSON):\n",
        "    generate_mock_coco_json(DATASET_JSON)"
    ]
})

# Cell 5: Run EDA (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "with open(DATASET_JSON, 'r') as f:\n",
        "    coco = json.load(f)\n",
        "images = coco[\"images\"]\n",
        "annotations = coco[\"annotations\"]\n",
        "categories = {cat[\"id\"]: cat[\"name\"] for cat in coco[\"categories\"]}\n",
        "\n",
        "# Categories\n",
        "cat_counts = {}\n",
        "for ann in annotations:\n",
        "    cat_counts[categories[ann[\"category_id\"]]] = cat_counts.get(categories[ann[\"category_id\"]], 0) + 1\n",
        "\n",
        "# BBox areas\n",
        "areas = [ann[\"bbox\"][2] * ann[\"bbox\"][3] for ann in annotations]\n",
        "altitudes = [img[\"altitude\"] for img in images if \"altitude\" in img]\n",
        "\n",
        "# Display distribution\n",
        "print(f\"Frames Analyzed: {len(images)}\")\n",
        "print(f\"Total Targets: {len(annotations)}\")\n",
        "print(\"\\nCategory Counts:\")\n",
        "for k, v in cat_counts.items():\n",
        "    print(f\"  * {k}: {v} ({v/len(annotations)*100:.1f}%)\")\n",
        "\n",
        "# Plot scale histogram\n",
        "plt.figure(figsize=(10, 4))\n",
        "plt.hist(np.sqrt(areas), bins=30, color='skyblue', edgecolor='black')\n",
        "plt.title(\"Target Scale Distribution (sqrt BBox Area in Pixels)\")\n",
        "plt.xlabel(\"Equivalent Scale (px)\")\n",
        "plt.ylabel(\"Instances\")\n",
        "plt.show()"
    ]
})

# Cell 6: EDL Section (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Part 2: Evidential Deep Learning (EDL) Classifier Pipeline\n",
        "Here we define our PyTorch-based training pipeline using Softplus evidence layers and Dirichlet NLL loss. We also implement ECE metrics to check model calibration."
    ]
})

# Cell 7: EDL Pipeline Code (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Torch availability guard ──────────────────────────────────────────────\n",
        "try:\n",
        "    import torch\n",
        "    import torch.nn as nn\n",
        "    import torch.optim as optim\n",
        "    import torch.nn.functional as F          # FIX: import F correctly\n",
        "    from torch.utils.data import Dataset, DataLoader\n",
        "    TORCH_AVAILABLE = True\n",
        "    print(f\"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}\")\n",
        "except ImportError:\n",
        "    TORCH_AVAILABLE = False\n",
        "    print(\"WARNING: torch not found. Run: uv add torch torchvision\")\n",
        "\n",
        "\n",
        "def calculate_ece(probs, labels, num_bins=10):\n",
        "    \"\"\"Expected Calibration Error over `num_bins` confidence bins.\"\"\"\n",
        "    bin_boundaries = np.linspace(0, 1, num_bins + 1)\n",
        "    ece = 0.0\n",
        "    pred_probs  = np.max(probs, axis=1)\n",
        "    pred_labels = np.argmax(probs, axis=1)\n",
        "    for i in range(num_bins):\n",
        "        in_bin = (pred_probs > bin_boundaries[i]) & (pred_probs <= bin_boundaries[i + 1])\n",
        "        prop = np.mean(in_bin)\n",
        "        if prop > 0:\n",
        "            acc  = np.mean(pred_labels[in_bin] == labels[in_bin])\n",
        "            conf = np.mean(pred_probs[in_bin])\n",
        "            ece += prop * np.abs(acc - conf)\n",
        "    return ece\n",
        "\n",
        "\n",
        "if TORCH_AVAILABLE:\n",
        "    class SeaDronesSeeDataset(Dataset):\n",
        "        \"\"\"\n",
        "        Wraps SeaDronesSee COCO annotations into a PyTorch Dataset.\n",
        "        Args:\n",
        "            annotations : list  -- COCO annotation records\n",
        "            images      : dict  -- {image_id: image_record}  <-- must be a dict!\n",
        "            cat_to_idx  : dict  -- {category_id: class_index}\n",
        "        \"\"\"\n",
        "        def __init__(self, annotations, images, cat_to_idx):\n",
        "            self.features = []\n",
        "            self.labels   = []\n",
        "            for ann in annotations:\n",
        "                img      = images[ann[\"image_id\"]]   # images MUST be a dict\n",
        "                bbox     = ann[\"bbox\"]\n",
        "                w_norm   = bbox[2] / img[\"width\"]\n",
        "                h_norm   = bbox[3] / img[\"height\"]\n",
        "                aspect   = bbox[2] / (bbox[3] + 1e-6)\n",
        "                cx_norm  = (bbox[0] + bbox[2] / 2.0) / img[\"width\"]\n",
        "                cy_norm  = (bbox[1] + bbox[3] / 2.0) / img[\"height\"]\n",
        "                rel_area = (bbox[2] * bbox[3]) / (img[\"width\"] * img[\"height\"])  # FIX: no stray paren\n",
        "                self.features.append([w_norm, h_norm, aspect, cx_norm, cy_norm, rel_area])\n",
        "                self.labels.append(cat_to_idx[ann[\"category_id\"]])\n",
        "            self.features = torch.tensor(self.features, dtype=torch.float32)\n",
        "            self.labels   = torch.tensor(self.labels,   dtype=torch.long)\n",
        "        def __len__(self):          return len(self.labels)\n",
        "        def __getitem__(self, idx): return self.features[idx], self.labels[idx]\n",
        "\n",
        "    class EDLClassifier(nn.Module):\n",
        "        \"\"\"Evidential Deep Learning classifier with Softplus evidence output.\"\"\"\n",
        "        def __init__(self, input_dim, num_classes):\n",
        "            super().__init__()\n",
        "            self.net = nn.Sequential(\n",
        "                nn.Linear(input_dim, 64),\n",
        "                nn.ReLU(),\n",
        "                nn.Linear(64, 32),\n",
        "                nn.ReLU(),\n",
        "                nn.Linear(32, num_classes),\n",
        "            )\n",
        "        def forward(self, x):\n",
        "            return F.softplus(self.net(x))  # FIX: use the imported F alias\n",
        "\n",
        "    def edl_loss(alpha, y_onehot, epoch_idx, num_classes):\n",
        "        \"\"\"\n",
        "        Evidential Dirichlet NLL + annealed KL regulariser.\n",
        "        Ref: Sensoy et al. NeurIPS 2018.\n",
        "        \"\"\"\n",
        "        S             = torch.sum(alpha, dim=1, keepdim=True)           # (B, 1)\n",
        "        loss_nll      = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)  # (B,)\n",
        "        alp_tilde     = y_onehot + (1.0 - y_onehot) * alpha            # remove gt evidence\n",
        "        kl_alpha      = torch.ones((1, num_classes), device=alpha.device)\n",
        "        sum_alp_tilde = torch.sum(alp_tilde, dim=1, keepdim=True)      # (B, 1)\n",
        "        first_term    = torch.lgamma(sum_alp_tilde) - torch.lgamma(torch.sum(kl_alpha, dim=1, keepdim=True))\n",
        "        second_term   = torch.sum(torch.lgamma(kl_alpha) - torch.lgamma(alp_tilde), dim=1, keepdim=True)\n",
        "        third_term    = torch.sum((alp_tilde - 1.0) * (torch.digamma(alp_tilde) - torch.digamma(sum_alp_tilde)), dim=1, keepdim=True)\n",
        "        # FIX: reshape(-1) instead of squeeze() -- squeeze() collapses to scalar when batch=1\n",
        "        loss_kl       = (first_term + second_term + third_term).reshape(-1)  # (B,)\n",
        "        beta          = min(1.0, epoch_idx / 10)   # KL annealing schedule\n",
        "        return torch.mean(loss_nll + beta * loss_kl)"
    ]
})

# Cell 8: Run EDL Training (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "img_dict = {img[\"id\"]: img for img in images}\n",
        "cat_to_idx = {cid: idx for idx, cid in enumerate(sorted(categories.keys()))}\n",
        "dataset = SeaDronesSeeDataset(annotations, img_dict, cat_to_idx)\n",
        "train_loader = DataLoader(dataset, batch_size=16, shuffle=True)\n",
        "\n",
        "model = EDLClassifier(input_dim=6, num_classes=len(categories))\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.01)\n",
        "\n",
        "print(\"Training Evidential Deep Learning Classifier...\")\n",
        "for epoch in range(1, 6):\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "    for x, y in train_loader:\n",
        "        optimizer.zero_grad()\n",
        "        evidence = model(x)\n",
        "        alpha = evidence + 1.0\n",
        "        y_onehot = torch.nn.functional.one_hot(y, num_classes=len(categories)).float()\n",
        "        loss = edl_loss(alpha, y_onehot, epoch, len(categories))\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item() * len(x)\n",
        "        \n",
        "    # Get training calibration\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        evidence = model(dataset.features)\n",
        "        probs = (evidence + 1.0) / torch.sum(evidence + 1.0, dim=1, keepdim=True)\n",
        "        val_ece = calculate_ece(probs.numpy(), dataset.labels.numpy())\n",
        "        acc = np.mean(np.argmax(probs.numpy(), axis=1) == dataset.labels.numpy())\n",
        "        \n",
        "    print(f\"Epoch {epoch}/5 | Loss: {total_loss/len(dataset):.4f} | Acc: {acc:.3f} | ECE: {val_ece:.4f}\")"
    ]
})

# Cell 9: Simulation Section (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Part 3: Active Evidential Sensing & Rescue Routing Simulation\n",
        "This section implements the spatial coordinate projection, state-dependent Kalman Filter tracking, and the logistics-aware Risk-UCB priority engine."
    ]
})

# Cell 10: Simulator Code (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class EvidentialClassifierSimulator:\n",
        "    def __init__(self, num_classes=4):\n",
        "        self.num_classes = num_classes\n",
        "        self.risk_weights = np.array([1.0, 0.4, 0.2, 0.1])\n",
        "    def estimate(self, true_class, distance, occluded=False):\n",
        "        evidence = np.zeros(self.num_classes)\n",
        "        if occluded:\n",
        "            evidence += np.random.uniform(0.05, 0.2, self.num_classes)\n",
        "        else:\n",
        "            max_ev = max(0.5, 12.0 - 0.05 * distance)\n",
        "            evidence += 0.05\n",
        "            evidence[true_class] += max_ev\n",
        "            evidence += np.random.uniform(0.0, 0.1, self.num_classes)\n",
        "        alpha = evidence + 1.0\n",
        "        S = np.sum(alpha)\n",
        "        return evidence / S, self.num_classes / S, alpha / S\n",
        "\n",
        "class UncertaintyKalmanFilter:\n",
        "    def __init__(self, dt=1.0):\n",
        "        self.dt = dt\n",
        "        self.F = np.array([\n",
        "            [1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],\n",
        "            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]\n",
        "        ])\n",
        "        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])\n",
        "        self.Q = np.eye(4) * 1e-5\n",
        "        self.P = np.eye(4) * 1.0\n",
        "    def predict(self, state, current_drift):\n",
        "        drift = np.array([0.0, 0.0, current_drift[0], current_drift[1]])\n",
        "        self.P = self.F @ self.P @ self.F.T + self.Q\n",
        "        return self.F @ state + drift\n",
        "    def update(self, state_pred, measurement, spatial_cov, epistemic_unc, gamma=2.0):\n",
        "        R = spatial_cov + gamma * epistemic_unc * np.eye(2)\n",
        "        S_cov = self.H @ self.P @ self.H.T + R\n",
        "        K = self.P @ self.H.T @ np.linalg.inv(S_cov)\n",
        "        self.P = (np.eye(4) - K @ self.H) @ self.P\n",
        "        return state_pred + K @ (measurement - self.H @ state_pred)"
    ]
})

# Cell 11: Run Simulation (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "sim_classifier = EvidentialClassifierSimulator()\n",
        "uav_pos = np.array([0.0, 0.0])\n",
        "uav_speed = 6.0\n",
        "ocean_current = np.array([0.1, -0.05])\n",
        "victims = {\n",
        "    1: {\"state\": np.array([60.0, 45.0, 0.0, 0.0]), \"class\": 0, \"occluded\": True, \"decay\": 0.08, \"name\": \"Victim A (Drowning, Occluded)\"},\n",
        "    2: {\"state\": np.array([30.0, -20.0, 0.0, 0.0]), \"class\": 2, \"occluded\": False, \"decay\": 0.02, \"name\": \"Victim B (Swimming, Clear)\"}\n",
        "}\n",
        "trackers = {i: UncertaintyKalmanFilter() for i in victims}\n",
        "states = {i: victims[i][\"state\"].copy() for i in victims}\n",
        "\n",
        "print(\"Starting Closed-Loop AES-RARR Simulation Routing...\")\n",
        "for step in range(1, 6):\n",
        "    print(f\"\\n--- Step {step} | UAV: ({uav_pos[0]:.1f}, {uav_pos[1]:.1f}) ---\")\n",
        "    priorities = {}\n",
        "    target_positions = {}\n",
        "    for vid, data in victims.items():\n",
        "        data[\"state\"][:2] += ocean_current + np.random.normal(0, 0.1, 2)\n",
        "        true_pos = data[\"state\"][:2]\n",
        "        dist = np.linalg.norm(true_pos - uav_pos)\n",
        "        \n",
        "        spatial_cov = np.eye(2) * ((0.5 + 0.01 * dist) ** 2)\n",
        "        meas = true_pos + np.random.multivariate_normal([0, 0], spatial_cov)\n",
        "        \n",
        "        beliefs, u, probs = sim_classifier.estimate(data[\"class\"], dist, data[\"occluded\"])\n",
        "        states[vid] = trackers[vid].update(trackers[vid].predict(states[vid], ocean_current), meas, spatial_cov, u)\n",
        "        \n",
        "        exp_risk = np.sum(probs * sim_classifier.risk_weights)\n",
        "        p_raw = exp_risk + 0.8 * u * (1.0 - exp_risk)\n",
        "        travel_time = dist / uav_speed\n",
        "        priority_score = (p_raw * np.exp(data[\"decay\"] * travel_time)) / (dist + 1.0)\n",
        "        \n",
        "        priorities[vid] = priority_score\n",
        "        target_positions[vid] = states[vid][:2]\n",
        "        print(f\"  Target {vid} ({data['name'][:10]}): Dist={dist:.1f}m | Uncertainty={u:.3f} | Priority={priority_score:.3f}\")\n",
        "        \n",
        "    best_target = max(priorities, key=priorities.get)\n",
        "    print(f\"  ==> Active Allocation: Flying toward Target {best_target} ({victims[best_target]['name']})\")\n",
        "    direction = target_positions[best_target] - uav_pos\n",
        "    d_norm = np.linalg.norm(direction)\n",
        "    if d_norm <= uav_speed:\n",
        "        uav_pos = target_positions[best_target].copy()\n",
        "        victims[best_target][\"occluded\"] = False\n",
        "    else:\n",
        "        uav_pos += (direction / d_norm) * uav_speed"
    ]
})

# Compile the notebook data structure into JSON format
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write the Jupyter Notebook JSON file
output_notebook_path = "d:/dev/DAP/maritime_sar_demo.ipynb"
with open(output_notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated Jupyter Notebook at: {output_notebook_path}")
