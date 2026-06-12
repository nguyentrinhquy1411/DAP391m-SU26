# Maritime SAR — AES-RARR Framework

> **Uncertainty-Aware Multi-Victim Prioritization for Maritime Search and Rescue**

---

## Project Structure

```
DAP/
├── pyproject.toml          # uv project manifest
├── README.md
├── .venv/                  # managed by uv (gitignored)
│
├── data/                   # SeaDronesSee annotations and dataset files
│   └── annotations/        # COCO val/train JSON files
│
├── src/                    # Reusable framework logic and Python package
│   ├── eda.py              # Exploratory data analysis implementation
│   ├── dataset.py          # PyTorch SeaDronesSee dataset wrapper
│   ├── models.py           # EDL model classes, Dirichlet loss, ECE metrics
│   ├── tracker.py          # uncertainty-scaled Kalman Filter target tracker
│   ├── simulator.py        # Evidential classifier simulators
│   └── simulation.py       # Multi-victim active sensing routing simulation
│
├── notebook/               # Modular Jupyter notebooks
│   ├── 01_eda.ipynb        # Part 1: Exploratory Data Analysis
│   ├── 02_edl_training.ipynb # Part 2: EDL Model Training
│   └── 03_tracking_prioritization.ipynb # Part 3: Rescue Routing Simulation
│
├── paper/                  # Academic paper and compilation resources
│   ├── maritime_sar.tex    # LaTeX source
│   ├── maritime_sar.pdf    # Compiled PDF
│   ├── pipeline.png        # Framework pipeline diagram
│   └── compile_latex.py    # Paper compilation script
│
├── models/                 # Model weight artifact files
│   ├── edl_weights.pth     # Trained PyTorch weights
│   └── edl_model_weights.npz # NumPy fallback weights
│
├── docs/                   # General documentation files
│   ├── edl_inputs_outputs.md # Evidential network feature specification
│   └── eda_summary_report.txt # Exploratory analysis summary report
│
├── eda_seadronessee.py     # Root CLI wrapper for EDA pipeline
├── training_pipeline.py    # Root CLI wrapper for EDL training (PyTorch/NumPy)
├── test_framework.py       # Root CLI wrapper for routing simulation
├── generate_notebook.py    # Recompiles all root and modular notebooks
└── maritime_sar_demo.ipynb  # Main end-to-end interactive demo notebook
```

---

## Quick Start with `uv`

### 1 — Install uv (if not already)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2 — Create environment and install all dependencies

```powershell
uv sync
```

### 3 — Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4 — Run the demo notebook

```powershell
uv run jupyter lab maritime_sar_demo.ipynb
```

### 5 — Run EDA pipeline

```powershell
uv run python eda_seadronessee.py
```

### 6 — Train EDL model

```powershell
uv run python training_pipeline.py
```

### 7 — Regenerate notebook

```powershell
uv run python generate_notebook.py
```

---

## Common `uv` Commands

| Task | Command |
|------|---------|
| Install all deps | `uv sync` |
| Add a new package | `uv add <package>` |
| Add a dev-only package | `uv add --dev <package>` |
| Remove a package | `uv remove <package>` |
| Run a script | `uv run python <script.py>` |
| Run Jupyter | `uv run jupyter lab` |
| Show installed packages | `uv pip list` |
| Export requirements | `uv pip freeze > requirements.txt` |
| Update all packages | `uv sync --upgrade` |

---

## Framework: AES-RARR

**Active Evidential Sensing + Risk-Averse Rescue Routing**

| Module | Description |
|--------|-------------|
| **EDL Detector** | YOLOv8 + Evidential head → aleatoric & epistemic uncertainty |
| **ECF Tracker** | Evidential Kalman Filter, `R ∝ u_epistemic` |
| **Risk Scorer** | Composite score: vitals × exposure × time-criticality |
| **U-Aware Router** | OR-Tools VRP solver with uncertainty-penalised costs |

---

## Citation

If you use this framework, please cite:

```bibtex
@article{aes_rarr_2024,
  title   = {Uncertainty-Aware Multi-Victim Prioritization for Maritime SAR},
  author  = {DAP Research Team},
  year    = {2024},
  journal = {arXiv preprint}
}
```
