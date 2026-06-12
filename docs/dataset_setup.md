# SeaDronesSee Dataset Setup Guide

This guide explains how to download the SeaDronesSee dataset from Kaggle and set it up in your local `data/` directory.

## Prerequisites

1. Create a Kaggle account if you don't have one.
2. Go to your Kaggle Account settings and click on **"Create New API Token"** to download `kaggle.json`.
3. Place `kaggle.json` in the location required by the Kaggle API (e.g., `C:\Users\<Windows-username>\.kaggle\kaggle.json` on Windows or `~/.kaggle/kaggle.json` on macOS/Linux).
4. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

## Downloading the Dataset

Run the following command in the root of the project to download the dataset directly into the `data/` directory:

```bash
# Ensure you use the correct Kaggle dataset ID for SeaDronesSee
kaggle datasets download -d <kaggle-dataset-id> -p data/ --unzip
```

Alternatively, you can manually download the dataset zip file from Kaggle, and extract its contents into the `data/` folder so that it matches the structure expected by the code (e.g., `data/annotations/instances_val.json`).

*Note: The `data/` directory is ignored by Git to prevent committing large dataset files to the repository.*
