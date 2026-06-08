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
├── eda_seadronessee.py     # SeaDronesSee EDA pipeline
├── training_pipeline.py    # EDL model training (PyTorch + NumPy fallback)
├── test_framework.py       # AES-RARR routing & tracking simulation
├── generate_notebook.py    # Programmatic Jupyter notebook generator
├── compile_latex.py        # LaTeX → PDF via YtoTech API
│
├── maritime_sar.tex         # Academic paper (LaTeX source)
├── maritime_sar.pdf         # Compiled paper
└── maritime_sar_demo.ipynb  # Interactive demo notebook
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
