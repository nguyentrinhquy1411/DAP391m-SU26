import os
import json

def create_notebook_file(cells, output_path):
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
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully generated notebook at: {output_path}")

def generate_all_notebooks():
    # -------------------------------------------------------------------------
    # 1. ROOT MASTER DEMO NOTEBOOK: maritime_sar_demo.ipynb
    # -------------------------------------------------------------------------
    master_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Active Evidential Sensing and Risk-Averse Rescue Routing (AES-RARR)\n",
                "### Maritime Search and Rescue AI Research Assistant\n",
                "\n",
                "This notebook provides a complete functional demonstration of the **AES-RARR** framework:\n",
                "1. **Exploratory Data Analysis (EDA):** Analyzes the SeaDronesSee annotation distributions and UAV telemetry.\n",
                "2. **Evidential Deep Learning (EDL) Pipeline:** Implements a Dirichlet-based classifier that outputs posture belief masses alongside epistemic uncertainty, including Expected Calibration Error (ECE) monitoring.\n",
                "3. **Closed-Loop Tracker & Prioritization Simulator:** Updates target state parameters under ocean current drift using a dynamic, uncertainty-scaled Kalman Filter, and routes rescue resources via a Risk-UCB priority engine.\n",
                "\n",
                "**Restructured Architecture:** This notebook imports core classes and logic directly from `/src` for reuse."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add root directory to sys.path to allow imports from src\n",
                "sys.path.append(os.path.abspath('.'))\n",
                "\n",
                "print(\"Libraries loaded and path configured.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 1: Exploratory Data Analysis (EDA)\n",
                "We parse the SeaDronesSee COCO JSON annotations. If the dataset isn't downloaded, we generate a mock dataset to allow dry-run execution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.eda import run_eda\n",
                "\n",
                "local_paths = [\n",
                "    \"./data/annotations/instances_val.json\",\n",
                "    \"./archive/compressed/annotations/instances_val.json\",\n",
                "    \"./archive/annotations/instances_val.json\",\n",
                "    \"./sds-dataset/annotations/instances_val.json\",\n",
                "    \"/content/sds-dataset/annotations/instances_val.json\"\n",
                "]\n",
                "\n",
                "DATASET_JSON = None\n",
                "for path in local_paths:\n",
                "    if os.path.exists(path):\n",
                "        DATASET_JSON = os.path.abspath(path)\n",
                "        print(f\"Found SeaDronesSee annotations at: {DATASET_JSON}\")\n",
                "        break\n",
                "\n",
                "if DATASET_JSON is None:\n",
                "    # Fallback / create mock in default /data path\n",
                "    DATASET_JSON = \"./data/annotations/instances_val.json\"\n",
                "    from src.eda import generate_mock_coco_json\n",
                "    generate_mock_coco_json(DATASET_JSON)\n",
                "\n",
                "run_eda(DATASET_JSON)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot scale histogram\n",
                "with open(DATASET_JSON, 'r') as f:\n",
                "    coco = json.load(f)\n",
                "areas = [ann[\"bbox\"][2] * ann[\"bbox\"][3] for ann in coco[\"annotations\"]]\n",
                "\n",
                "plt.figure(figsize=(10, 4))\n",
                "plt.hist(np.sqrt(areas), bins=30, color='skyblue', edgecolor='black')\n",
                "plt.title(\"Target Scale Distribution (sqrt BBox Area in Pixels)\")\n",
                "plt.xlabel(\"Equivalent Scale (px)\")\n",
                "plt.ylabel(\"Instances\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 2: Evidential Deep Learning (EDL) Classifier Pipeline\n",
                "Here we define our PyTorch-based training pipeline using Softplus evidence layers and Dirichlet NLL loss. We also implement ECE metrics to check model calibration."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import DataLoader\n",
                "\n",
                "from src.dataset import SeaDronesSeeDataset\n",
                "from src.models import EDLClassifier, edl_loss, calculate_ece\n",
                "\n",
                "categories = {cat[\"id\"]: cat[\"name\"] for cat in coco[\"categories\"]}\n",
                "img_dict = {img[\"id\"]: img for img in coco[\"images\"]}\n",
                "cat_to_idx = {cid: idx for idx, cid in enumerate(sorted(categories.keys()))}\n",
                "\n",
                "# Note: original notebook evaluates 6 geometric features (include_center_dist=False)\n",
                "dataset = SeaDronesSeeDataset(coco[\"annotations\"], img_dict, cat_to_idx, include_center_dist=False)\n",
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
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 2b: Q1 — EDL vs Softmax Calibration Comparison\n",
                "We train an identical softmax baseline and compare **Expected Calibration Error (ECE)** to answer *Q1: Does EDL produce better-calibrated uncertainty than deterministic softmax?*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.models import SoftmaxClassifier\n",
                "import pandas as pd\n",
                "\n",
                "softmax_model = SoftmaxClassifier(input_dim=6, num_classes=len(categories))\n",
                "sm_optimizer = optim.Adam(softmax_model.parameters(), lr=0.01)\n",
                "ce_loss_fn = torch.nn.CrossEntropyLoss()\n",
                "\n",
                "print(\"Training Softmax Baseline ...\")\n",
                "for epoch in range(1, 6):\n",
                "    softmax_model.train()\n",
                "    for x, y in train_loader:\n",
                "        sm_optimizer.zero_grad()\n",
                "        logits = softmax_model.net(x)\n",
                "        ce_loss_fn(logits, y).backward()\n",
                "        sm_optimizer.step()\n",
                "\n",
                "softmax_model.eval()\n",
                "with torch.no_grad():\n",
                "    sm_probs = softmax_model(dataset.features).numpy()\n",
                "    sm_ece   = calculate_ece(sm_probs, dataset.labels.numpy())\n",
                "    sm_acc   = float((sm_probs.argmax(1) == dataset.labels.numpy()).mean())\n",
                "\n",
                "# EDL final ECE evaluation\n",
                "model.eval()\n",
                "with torch.no_grad():\n",
                "    ev       = model(dataset.features)\n",
                "    edl_probs = (ev + 1.0) / (ev + 1.0).sum(1, keepdim=True)\n",
                "    edl_ece   = calculate_ece(edl_probs.numpy(), dataset.labels.numpy())\n",
                "    edl_acc   = float((edl_probs.numpy().argmax(1) == dataset.labels.numpy()).mean())\n",
                "\n",
                "q1_df = pd.DataFrame({\n",
                "    \"Model\":     [\"Softmax Baseline\", \"EDL (Ours)\"],\n",
                "    \"Accuracy\":  [f\"{sm_acc:.3f}\",   f\"{edl_acc:.3f}\"],\n",
                "    \"ECE (lower=better)\":     [f\"{sm_ece:.4f}\",   f\"{edl_ece:.4f}\"],\n",
                "    \"Calibrated?\": [\"\u2717\" if sm_ece > edl_ece else \"\u2713\",\n",
                "                    \"\u2713\" if edl_ece < sm_ece  else \"\u2717\"],\n",
                "})\n",
                "print(\"\\n=== Q1 Result: Calibration Comparison ===\")\n",
                "print(q1_df.to_string(index=False))\n",
                "print(f\"\\nEDL ECE improvement: {(sm_ece - edl_ece)/sm_ece*100:.1f}% lower than softmax\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 3: Active Evidential Sensing & Rescue Routing Simulation\n",
                "This section implements the spatial coordinate projection, state-dependent Kalman Filter tracking, and the logistics-aware Risk-UCB priority engine."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.simulator import EvidentialClassifierSimulator\n",
                "from src.tracker import UncertaintyKalmanFilter\n",
                "\n",
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
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 3b: Comparative Analysis — Comparative Simulation with Baselines\n",
                "We run the same scenario under **three modes** and compare **Expected Calibration Error** & **Victim Survival Rate**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.simulation import run_simulation\n",
                "\n",
                "print(\"Running comparative simulations (seed=42, 30 steps, 3 victims)...\")\n",
                "np.random.seed(42); rows_aes,  ba    = run_simulation(\"aes_rarr\")\n",
                "np.random.seed(42); rows_det,  bd    = run_simulation(\"deterministic\")\n",
                "np.random.seed(42); rows_dist, bdist = run_simulation(\"distance_router\")\n",
                "\n",
                "df = pd.DataFrame(rows_aes + rows_det + rows_dist)\n",
                "\n",
                "print(\"\\n\" + \"=\"*70)\n",
                "print(\"Risk-UCB vs Baselines: Per-Victim TTR & VSR\")\n",
                "print(\"=\"*70)\n",
                "print(df.to_string(index=False))\n",
                "\n",
                "# Aggregate summaries\n",
                "agg = (df.groupby(\"Mode\")[\"VSR\"].mean()\n",
                "         .reset_index()\n",
                "         .rename(columns={\"VSR\": \"Mean VSR\"})\n",
                "         .sort_values(\"Mean VSR\", ascending=False))\n",
                "rescued = (df.groupby(\"Mode\")[\"Rescued\"]\n",
                "             .apply(lambda s: (s == \"Yes\").sum())\n",
                "             .reset_index()\n",
                "             .rename(columns={\"Rescued\": \"Victims Rescued\"}))\n",
                "summary = agg.merge(rescued, on=\"Mode\")\n",
                "print(\"\\n-- Aggregate Summary ------------------------------------------------\")\n",
                "print(summary.to_string(index=False))"
            ]
        }
    ]
    
    create_notebook_file(master_cells, "maritime_sar_demo.ipynb")

    # -------------------------------------------------------------------------
    # 2. SMALL NOTEBOOK 1: notebook/01_eda.ipynb
    # -------------------------------------------------------------------------
    eda_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SeaDronesSee Exploratory Data Analysis (EDA)\n",
                "This notebook parses the SeaDronesSee COCO JSON dataset annotations to analyze drone altitude distribution, gimbal pitch, and target bounding box scales."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add parent folder to path to allow importing from src\n",
                "sys.path.append(os.path.abspath(os.path.join('..')))\n",
                "\n",
                "from src.eda import run_eda, generate_mock_coco_json"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "local_paths = [\n",
                "    \"../data/annotations/instances_val.json\",\n",
                "    \"../archive/compressed/annotations/instances_val.json\",\n",
                "    \"../archive/annotations/instances_val.json\",\n",
                "    \"../sds-dataset/annotations/instances_val.json\"\n",
                "]\n",
                "\n",
                "DATASET_JSON = None\n",
                "for path in local_paths:\n",
                "    if os.path.exists(path):\n",
                "        DATASET_JSON = os.path.abspath(path)\n",
                "        break\n",
                "\n",
                "if DATASET_JSON is None:\n",
                "    DATASET_JSON = \"../data/annotations/instances_val.json\"\n",
                "    generate_mock_coco_json(DATASET_JSON)\n",
                "\n",
                "print(f\"Using dataset: {DATASET_JSON}\")\n",
                "run_eda(DATASET_JSON)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(DATASET_JSON, 'r') as f:\n",
                "    coco = json.load(f)\n",
                "areas = [ann[\"bbox\"][2] * ann[\"bbox\"][3] for ann in coco[\"annotations\"]]\n",
                "\n",
                "plt.figure(figsize=(10, 4))\n",
                "plt.hist(np.sqrt(areas), bins=30, color='skyblue', edgecolor='black')\n",
                "plt.title(\"Target Scale Distribution (sqrt BBox Area in Pixels)\")\n",
                "plt.xlabel(\"Equivalent Scale (px)\")\n",
                "plt.ylabel(\"Instances\")\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook_file(eda_cells, "notebook/01_eda.ipynb")

    # -------------------------------------------------------------------------
    # 3. SMALL NOTEBOOK 2: notebook/02_edl_training.ipynb
    # -------------------------------------------------------------------------
    training_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Evidential Deep Learning (EDL) Classifier Pipeline\n",
                "This notebook trains a Dirichlet-based classifier under Evidential Deep Learning (EDL) and compares it with a deterministic Softmax baseline to inspect Expected Calibration Error (ECE)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import torch\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import DataLoader\n",
                "\n",
                "# Add parent folder to path to allow importing from src\n",
                "sys.path.append(os.path.abspath(os.path.join('..')))\n",
                "\n",
                "from src.dataset import SeaDronesSeeDataset\n",
                "from src.models import EDLClassifier, SoftmaxClassifier, edl_loss, calculate_ece\n",
                "from src.eda import generate_mock_coco_json"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "local_paths = [\n",
                "    \"../data/annotations/instances_val.json\",\n",
                "    \"../archive/compressed/annotations/instances_val.json\",\n",
                "    \"../archive/annotations/instances_val.json\",\n",
                "    \"../sds-dataset/annotations/instances_val.json\"\n",
                "]\n",
                "\n",
                "DATASET_JSON = None\n",
                "for path in local_paths:\n",
                "    if os.path.exists(path):\n",
                "        DATASET_JSON = os.path.abspath(path)\n",
                "        break\n",
                "\n",
                "if DATASET_JSON is None:\n",
                "    DATASET_JSON = \"../data/annotations/instances_val.json\"\n",
                "    generate_mock_coco_json(DATASET_JSON)\n",
                "\n",
                "with open(DATASET_JSON, 'r') as f:\n",
                "    coco = json.load(f)\n",
                "\n",
                "categories = {cat[\"id\"]: cat[\"name\"] for cat in coco[\"categories\"]}\n",
                "img_dict = {img[\"id\"]: img for img in coco[\"images\"]}\n",
                "cat_to_idx = {cid: idx for idx, cid in enumerate(sorted(categories.keys()))}\n",
                "\n",
                "dataset = SeaDronesSeeDataset(coco[\"annotations\"], img_dict, cat_to_idx, include_center_dist=False)\n",
                "train_loader = DataLoader(dataset, batch_size=16, shuffle=True)\n",
                "print(f\"Loaded dataset with {len(dataset)} samples.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
                "    model.eval()\n",
                "    with torch.no_grad():\n",
                "        evidence = model(dataset.features)\n",
                "        probs = (evidence + 1.0) / torch.sum(evidence + 1.0, dim=1, keepdim=True)\n",
                "        val_ece = calculate_ece(probs.numpy(), dataset.labels.numpy())\n",
                "        acc = np.mean(np.argmax(probs.numpy(), axis=1) == dataset.labels.numpy())\n",
                "        \n",
                "    print(f\"Epoch {epoch}/5 | Loss: {total_loss/len(dataset):.4f} | Acc: {acc:.3f} | ECE: {val_ece:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train softmax baseline\n",
                "softmax_model = SoftmaxClassifier(input_dim=6, num_classes=len(categories))\n",
                "sm_optimizer = optim.Adam(softmax_model.parameters(), lr=0.01)\n",
                "ce_loss_fn = torch.nn.CrossEntropyLoss()\n",
                "\n",
                "print(\"Training Softmax Baseline...\")\n",
                "for epoch in range(1, 6):\n",
                "    softmax_model.train()\n",
                "    for x, y in train_loader:\n",
                "        sm_optimizer.zero_grad()\n",
                "        logits = softmax_model.net(x)\n",
                "        ce_loss_fn(logits, y).backward()\n",
                "        sm_optimizer.step()\n",
                "\n",
                "softmax_model.eval()\n",
                "with torch.no_grad():\n",
                "    sm_probs = softmax_model(dataset.features).numpy()\n",
                "    sm_ece   = calculate_ece(sm_probs, dataset.labels.numpy())\n",
                "    sm_acc   = float((sm_probs.argmax(1) == dataset.labels.numpy()).mean())\n",
                "\n",
                "# EDL calibration evaluation\n",
                "model.eval()\n",
                "with torch.no_grad():\n",
                "    ev       = model(dataset.features)\n",
                "    edl_probs = (ev + 1.0) / (ev + 1.0).sum(1, keepdim=True)\n",
                "    edl_ece   = calculate_ece(edl_probs.numpy(), dataset.labels.numpy())\n",
                "    edl_acc   = float((edl_probs.numpy().argmax(1) == dataset.labels.numpy()).mean())\n",
                "\n",
                "q1_df = pd.DataFrame({\n",
                "    \"Model\":     [\"Softmax Baseline\", \"EDL (Ours)\"],\n",
                "    \"Accuracy\":  [f\"{sm_acc:.3f}\",   f\"{edl_acc:.3f}\"],\n",
                "    \"ECE (lower=better)\":     [f\"{sm_ece:.4f}\",   f\"{edl_ece:.4f}\"],\n",
                "    \"Calibrated?\": [\"\u2717\" if sm_ece > edl_ece else \"\u2713\",\n",
                "                    \"\u2713\" if edl_ece < sm_ece  else \"\u2717\"],\n",
                "})\n",
                "print(\"\\n=== Calibration Comparison ===\")\n",
                "print(q1_df.to_string(index=False))"
            ]
        }
    ]
    create_notebook_file(training_cells, "notebook/02_edl_training.ipynb")

    # -------------------------------------------------------------------------
    # 4. SMALL NOTEBOOK 3: notebook/03_tracking_prioritization.ipynb
    # -------------------------------------------------------------------------
    sim_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Closed-Loop Active Evidential Sensing & Rescue Routing Simulation\n",
                "This notebook runs closed-loop tracking simulations of targets drifted by ocean currents using a dynamic evidential Kalman Filter and schedules resource allocation using a Risk-UCB router."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "\n",
                "# Add parent folder to path to allow importing from src\n",
                "sys.path.append(os.path.abspath(os.path.join('..')))\n",
                "\n",
                "from src.simulator import EvidentialClassifierSimulator\n",
                "from src.tracker import UncertaintyKalmanFilter\n",
                "from src.simulation import run_simulation"
            ]
        },
        {
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
                "print(\"Starting Step-by-Step Simulation Routing...\")\n",
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
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Comparative Analysis: Risk-UCB vs Baselines\n",
                "We compare our full `aes_rarr` implementation against expected-risk `deterministic` and a `distance_router` first-nearest routing solver."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Running 30-step comparative simulation under seed=42...\")\n",
                "np.random.seed(42); rows_aes,  ba    = run_simulation(\"aes_rarr\")\n",
                "np.random.seed(42); rows_det,  bd    = run_simulation(\"deterministic\")\n",
                "np.random.seed(42); rows_dist, bdist = run_simulation(\"distance_router\")\n",
                "\n",
                "df = pd.DataFrame(rows_aes + rows_det + rows_dist)\n",
                "print(df.to_string(index=False))\n",
                "\n",
                "# Aggregate survival summaries\n",
                "agg = (df.groupby(\"Mode\")[\"VSR\"].mean()\n",
                "         .reset_index()\n",
                "         .rename(columns={\"VSR\": \"Mean VSR\"})\n",
                "         .sort_values(\"Mean VSR\", ascending=False))\n",
                "rescued = (df.groupby(\"Mode\")[\"Rescued\"]\n",
                "             .apply(lambda s: (s == \"Yes\").sum())\n",
                "             .reset_index()\n",
                "             .rename(columns={\"Rescued\": \"Victims Rescued\"}))\n",
                "summary = agg.merge(rescued, on=\"Mode\")\n",
                "print(\"\\n=== Comparative Summary ===\")\n",
                "print(summary.to_string(index=False))"
            ]
        }
    ]
    create_notebook_file(sim_cells, "notebook/03_tracking_prioritization.ipynb")

if __name__ == "__main__":
    generate_all_notebooks()
