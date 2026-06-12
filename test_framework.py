import os
import json
import numpy as np

from src.simulator import EvidentialClassifier
from src.tracker import UncertaintyKalmanFilter

# 1. Configuration for Kagglehub and Content Target Paths
DATASET_ID = "ubiratanfilho/sds-dataset"
TARGET_PATH = "/content/sds-dataset"


def load_actual_dataset():
    """
    Attempts to download and load annotations from SeaDronesSee using kagglehub.
    """
    print("\n" + "="*80)
    print("KAGGLEHUB DATASET LOADING & PARSING")
    print("="*80)
    
    # Check standard local paths first (prioritizing /data)
    local_paths = [
        "./data/annotations/instances_val.json",
        "./archive/compressed/annotations/instances_val.json",
        "./archive/annotations/instances_val.json",
        "./sds-dataset/annotations/instances_val.json"
    ]
    for path in local_paths:
        if os.path.exists(path):
            json_path = os.path.abspath(path)
            print(f"Found local SeaDronesSee dataset annotations at: {json_path}")
            print(f"Parsing COCO metadata from: {json_path}")
            with open(json_path, 'r') as f:
                return json.load(f)

    try:
        import kagglehub
        print(f"1. Downloading dataset '{DATASET_ID}' via kagglehub...")
        cache_path = kagglehub.dataset_download(DATASET_ID)
        print(f"   Cached location: {cache_path}")
        
        # Verify annotations directory
        ann_dir = os.path.join(cache_path, "annotations")
        if not os.path.exists(ann_dir):
            ann_dir = cache_path # fallback if direct zip contents differ
            
        json_path = os.path.join(ann_dir, "instances_val.json")
        if not os.path.exists(json_path):
            # Check for instances_train.json as secondary option
            json_path = os.path.join(ann_dir, "instances_train.json")
            
        if not os.path.exists(json_path):
            print(f"   [Warning] COCO json annotation file not found in {cache_path}.")
            return None
            
        print(f"2. Parsing COCO metadata from: {json_path}")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        return coco_data
    except ImportError:
        print("[System Info] 'kagglehub' package is not installed.")
        print("              Run: 'pip install kagglehub' to enable real dataset parsing.")
        return None
    except Exception as e:
        print(f"   [Warning] Dataset loading encountered an error: {e}")
        return None


def run_seadronessee_prioritization(coco_data):
    """
    Runs our uncertainty-aware scheduling using the downloaded SeaDronesSee annotations.
    """
    print("\n" + "="*80)
    print("RUNNING AES-RARR ON SEADRONESSEE DATASET")
    print("="*80)
    
    # Category mapping to Risk Weights
    # SeaDronesSee Categories: swimmer (highest), floater, boat, life jacket, buoy, etc.
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    print(f"Detected categories in dataset: {categories}")
    
    # Risk weights map based on rescue priority
    # swimmer=1.0 (drowning threat), floater=0.4 (PFD), life jacket=0.1, boat=0.2, buoy=0.05
    risk_mapping = {}
    for cid, name in categories.items():
        if "swimmer" in name.lower():
            risk_mapping[cid] = 1.0
        elif "floater" in name.lower():
            risk_mapping[cid] = 0.4
        elif "boat" in name.lower() or "jet ski" in name.lower():
            risk_mapping[cid] = 0.2
        elif "life jacket" in name.lower():
            risk_mapping[cid] = 0.1
        else:
            risk_mapping[cid] = 0.05
            
    images = coco_data["images"][:5] # Parse first 5 frames
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        
    uav_pos = np.array([0.0, 0.0])
    risk_aversion = 0.8
    classifier = EvidentialClassifier()
    
    for idx, img in enumerate(images):
        img_id = img["id"]
        file_name = img["file_name"]
        
        # Parse image-level telemetry if available, else use default values
        alt = img.get("altitude", 45.0) # default height 45m
        lat = img.get("latitude", 20.84)
        lon = img.get("longitude", 107.03)
        
        print(f"\nFrame: {file_name} | Target ID: {img_id}")
        print(f"UAV Telemetry -> Alt: {alt}m | GPS: ({lat:.4f}, {lon:.4f})")
        
        anns = annotations_by_image.get(img_id, [])
        if not anns:
            print("  No targets annotated in this frame.")
            continue
            
        print(f"  {'Class Name':<15} | {'BBox Centroid':<15} | {'Exp Risk':<8} | {'Priority':<8}")
        print("  " + "-"*60)
        
        for ann in anns:
            cat_id = ann["category_id"]
            class_name = categories.get(cat_id, "Unknown")
            bbox = ann["bbox"] # [x, y, width, height]
            
            # Compute visual centroid
            centroid = np.array([bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0])
            
            # Derive distance proxy (pixel distance from image center)
            img_center = np.array([img["width"]/2.0, img["height"]/2.0])
            pixel_distance = np.linalg.norm(centroid - img_center)
            
            # Map standard categories to postures for evidential classification
            risk_val = risk_mapping.get(cat_id, 0.05)
            true_class_idx = 0 if risk_val == 1.0 else (1 if risk_val == 0.4 else 2)
            
            beliefs, epistemic_unc, probs = classifier.estimate(
                true_class_idx, pixel_distance, occluded=False
            )
            
            # Expected Risk
            exp_risk = np.sum(probs * classifier.risk_weights)
            
            # Risk-UCB Prioritization
            priority_score = exp_risk + risk_aversion * epistemic_unc * (1.0 - exp_risk)
            
            print(f"  {class_name:<15} | ({centroid[0]:.1f}, {centroid[1]:.1f}) | {exp_risk:<8.3f} | {priority_score:<8.3f}")


def run_fallback_simulation():
    """Fallback simulation in case dataset or kagglehub are unavailable."""
    print("\n" + "="*80)
    print("RUNNING HIGH-FIDELITY SIMULATION (FALLBACK)")
    print("="*80)
    
    classifier = EvidentialClassifier()
    dt = 1.0
    uav_pos = np.array([0.0, 0.0])
    uav_speed = 5.0
    ocean_current = np.array([0.1, -0.05])
    
    victicks = {
        1: {"state": np.array([80.0, 60.0, 0.0, 0.0]), "class": 0, "occluded": True,  "decay": 0.08, "name": "Victim A (Drowning, Occluded)"},
        2: {"state": np.array([40.0, -30.0, 0.0, 0.0]), "class": 2, "occluded": False, "decay": 0.02, "name": "Victim B (Swimming, Clear)"},
        3: {"state": np.array([10.0, 20.0, 0.0, 0.0]), "class": 3, "occluded": False, "decay": 0.005, "name": "Victim C (PFD Floater)"}
    }
    
    trackers = {i: UncertaintyKalmanFilter(dt) for i in victicks}
    states = {i: victicks[i]["state"].copy() for i in victicks}
    
    for step in range(1, 4): # Run 3 quick steps for demonstration
        print(f"\n--- STEP {step} ---")
        print(f"UAV Position: ({uav_pos[0]:.2f}, {uav_pos[1]:.2f})")
        print(f"{'Victim ID & Name':<30} | {'Dist (m)':<8} | {'Epist. u':<8} | {'Exp Risk':<8} | {'Priority':<8}")
        print("-" * 80)
        
        priorities = {}
        target_positions = {}
        
        for vid, data in victicks.items():
            data["state"][:2] += ocean_current * dt + np.random.normal(0, 0.2, 2)
            true_pos = data["state"][:2]
            dist = np.linalg.norm(true_pos - uav_pos)
            
            # Measurement spatial covariance
            base_spatial_noise = 0.5 + 0.02 * dist
            spatial_cov = np.eye(2) * (base_spatial_noise ** 2)
            measurement = true_pos + np.random.multivariate_normal([0, 0], spatial_cov)
            
            # EDL update
            beliefs, epistemic_unc, probs = classifier.estimate(data["class"], dist, data["occluded"])
            
            # Tracking
            tracker = trackers[vid]
            state_pred = tracker.predict(states[vid], ocean_current)
            states[vid] = tracker.update(state_pred, measurement, spatial_cov, epistemic_unc, gamma=5.0)
            
            # PVI Score
            exp_risk = np.sum(probs * classifier.risk_weights)
            p_raw = exp_risk + 0.8 * epistemic_unc * (1.0 - exp_risk)
            travel_time = dist / uav_speed
            priority_score = (p_raw * np.exp(data["decay"] * travel_time)) / (dist + 1.0)
            
            priorities[vid] = priority_score
            target_positions[vid] = states[vid][:2]
            
            print(f"{vid}: {data['name'][:25]:<22} | {dist:<8.2f} | {epistemic_unc:<8.3f} | {exp_risk:<8.3f} | {priority_score:<8.3f}")
            
        best_target = max(priorities, key=priorities.get)
        print(f"--> UAV Active Allocation: Hovering toward Target {best_target} ({victicks[best_target]['name']})")
        uav_pos += ((target_positions[best_target] - uav_pos) / (np.linalg.norm(target_positions[best_target] - uav_pos) + 1e-5)) * uav_speed * dt


if __name__ == "__main__":
    coco_data = load_actual_dataset()
    if coco_data is not None:
        run_seadronessee_prioritization(coco_data)
    else:
        run_fallback_simulation()
