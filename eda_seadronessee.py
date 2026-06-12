import os
from src.eda import run_eda

# Configurations
local_paths = [
    "./data/annotations/instances_val.json",
    "./archive/compressed/annotations/instances_val.json",
    "./archive/annotations/instances_val.json",
    "./sds-dataset/annotations/instances_val.json",
    "/content/sds-dataset/annotations/instances_val.json"
]

DATASET_JSON = "./data/annotations/instances_val.json"
for path in local_paths:
    if os.path.exists(path):
        DATASET_JSON = os.path.abspath(path)
        break

if __name__ == "__main__":
    run_eda(DATASET_JSON)
