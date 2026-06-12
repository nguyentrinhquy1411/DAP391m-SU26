import os
import math
import numpy as np
from .tracker import UncertaintyKalmanFilter
from .simulator import EvidentialClassifierSimulator

# Try importing torch and image libraries
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Simulation defaults
UAV_SPEED_DEFAULT = 10.0
OCEAN_DRIFT_DEFAULT = np.array([0.1, -0.05])
TAU_UNC_DEFAULT = 0.55
LAMBDA_RA_DEFAULT = 0.8
GAMMA_R_DEFAULT = 2.0
SIM_STEPS_DEFAULT = 30
RESCUE_DIST_DEFAULT = 5.0
DECAY_RATES_DEFAULT = {0: 0.08, 1: 0.04, 2: 0.02, 3: 0.01}

VICTIM_INIT_DEFAULT = {
    1: {"pos": np.array([55.0,  40.0]), "class": 0, "occluded": True,
        "name": "Victim A (Drowning, Occluded)"},
    2: {"pos": np.array([25.0, -18.0]), "class": 2, "occluded": False,
        "name": "Victim B (Swimming, Clear)"},
    3: {"pos": np.array([-35.0, 28.0]), "class": 1, "occluded": False,
        "name": "Victim C (Floating)"},
}

_crop_cache = {}

def get_class_crop_sample(true_class, base_dir="data/images", annotations_path="data/annotations/instances_val.json"):
    """
    Caches and returns a random real crop from the SeaDronesSee validation set
    corresponding to the simulated victim's category class.
    """
    global _crop_cache
    if not TORCH_AVAILABLE:
        return None
        
    if not _crop_cache:
        # Load validation annotations and cache some samples
        if not os.path.exists(annotations_path):
            return None
        try:
            import json
            with open(annotations_path, 'r') as f:
                coco = json.load(f)
            images = {img["id"]: img for img in coco["images"]}
            
            # Resolve image paths
            resolved_paths = {}
            for img_id, img_info in images.items():
                fname = img_info["file_name"]
                for folder in ["val", "train", "test", ""]:
                    path = os.path.join(base_dir, folder, fname) if folder else os.path.join(base_dir, fname)
                    if os.path.exists(path):
                        resolved_paths[img_id] = os.path.abspath(path)
                        break
            
            # Group sample bboxes by category (1=swimmer, 2=floater, 4=life jacket)
            for cid in [1, 2, 4]:
                _crop_cache[cid] = []
                
            for ann in coco["annotations"]:
                cid = ann["category_id"]
                if cid in [1, 2, 4]:
                    img_id = ann["image_id"]
                    if img_id in resolved_paths and len(_crop_cache[cid]) < 20:
                        img_path = resolved_paths[img_id]
                        with Image.open(img_path) as img:
                            x, y, w, h = ann["bbox"]
                            x1 = max(0, int(round(x)))
                            y1 = max(0, int(round(y)))
                            x2 = min(img.width, int(round(x + w)))
                            y2 = min(img.height, int(round(y + h)))
                            if x2 > x1 and y2 > y1:
                                crop = img.crop((x1, y1, x2, y2)).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                                _crop_cache[cid].append(crop)
        except Exception as e:
            print(f"[Warning] Failed to build crop cache: {e}")
            
    # Map simulation class index (0-3) to SeaDronesSee category IDs (1, 2, 4)
    # 0 (Drowning) -> swimmer (1)
    # 1 (Floating) -> floater (2)
    # 2 (Swimming) -> swimmer (1)
    # 3 (PFD Floater) -> life jacket (4)
    sim_to_coco = {0: 1, 1: 2, 2: 1, 3: 4}
    cid = sim_to_coco.get(true_class, 1)
    samples = _crop_cache.get(cid, [])
    if samples:
        idx = np.random.randint(len(samples))
        return samples[idx]
    return None


def select_next_target_lookahead(uav_pos, unrescued_vids, vics, uav_speed, decay_rates, descent_latency, depth=3):
    """
    Finds the next target to visit using a lookahead rollout.
    Maximizes the joint expected survival probability of victims.
    """
    if not unrescued_vids:
        return None
        
    best_target = list(unrescued_vids)[0]
    best_val = -1.0
    
    path_len = min(depth, len(unrescued_vids))
    import itertools
    sequences = list(itertools.permutations(unrescued_vids, path_len))
    
    for seq in sequences:
        current_uav = uav_pos.copy()
        current_time = 0.0
        survival_probs = {}
        
        # Estimate survival probability for visited targets in sequence
        for vid in seq:
            vic_data = vics[vid]
            vic_pos = vic_data["state"][:2]
            dist = np.linalg.norm(vic_pos - current_uav)
            
            travel_time = dist / uav_speed
            # If target has high uncertainty, assume a descent is required
            if vic_data.get("occluded", False):
                travel_time += descent_latency
                
            current_time += travel_time
            decay = decay_rates[vic_data["class"]]
            survival_probs[vid] = math.exp(-decay * current_time)
            current_uav = vic_pos.copy()
            
        # Estimate survival probability for remaining unvisited targets
        for vid in unrescued_vids:
            if vid not in survival_probs:
                vic_data = vics[vid]
                vic_pos = vic_data["state"][:2]
                dist_from_last = np.linalg.norm(vic_pos - current_uav)
                extra_time = dist_from_last / uav_speed
                decay = decay_rates[vic_data["class"]]
                # Apply penalty for late rescue
                survival_probs[vid] = math.exp(-decay * (current_time + extra_time)) * 0.5
                
        total_vsr = sum(survival_probs.values())
        if total_vsr > best_val:
            best_val = total_vsr
            best_target = seq[0]
            
    return best_target


def run_simulation(mode: str, 
                   uav_speed=UAV_SPEED_DEFAULT, 
                   ocean_drift=OCEAN_DRIFT_DEFAULT, 
                   tau_unc=TAU_UNC_DEFAULT, 
                   lambda_ra=LAMBDA_RA_DEFAULT, 
                   gamma_r=GAMMA_R_DEFAULT, 
                   sim_steps=SIM_STEPS_DEFAULT, 
                   rescue_dist=RESCUE_DIST_DEFAULT, 
                   decay_rates=DECAY_RATES_DEFAULT, 
                   victim_init=VICTIM_INIT_DEFAULT,
                   descent_latency=1.0,
                   return_history=False):
    """
    Runs target tracking and UAV search & rescue routing simulation under five modes:
    'aes_rarr' | 'no_branch' | 'static_R' | 'deterministic' | 'distance_router'
    """
    clf = EvidentialClassifierSimulator()
    uav = np.array([0.0, 0.0])
    history = []
    vic_metrics = {vid: {"beliefs": np.array([0.25, 0.25, 0.25, 0.25]), "u": 1.0, "er": 0.5, "score": 0.0} for vid in victim_init}
    
    # Initialize victims
    vics = {vid: {
        "state":         np.array([d["pos"][0], d["pos"][1], 0.0, 0.0]),
        "class":         d["class"],
        "occluded":      d["occluded"],
        "name":          d["name"],
        "rescued":       False,
        "rescue_time":   None,
        "active_branch": False,
    } for vid, d in victim_init.items()}
    
    trs = {vid: UncertaintyKalmanFilter() for vid in vics}
    sts = {vid: vics[vid]["state"].copy() for vid in vics}
    branches = 0
    cumulative_time = 0.0
    total_descent_attempts = 0
    true_positive_descents = 0
    best_last = None

    # Load real Evidential CNN weights if torch is available
    cnn_model = None
    if TORCH_AVAILABLE:
        from src.models import EvidentialCNNClassifier
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = "models/edl_weights.pth"
        if os.path.exists(weights_path):
            try:
                cnn_model = EvidentialCNNClassifier(num_classes=5) # 5 categories
                cnn_model.load_state_dict(torch.load(weights_path, map_location=device))
                cnn_model.to(device)
                cnn_model.eval()
            except Exception as e:
                print(f"[Warning] Simulation failed to load EDL CNN weights ({e}). Using proxy.")
                cnn_model = None

    for step in range(1, sim_steps + 1):
        branch_step = False
        pris = {}
        tpos = {}
        cumulative_time += 1.0

        for vid, data in vics.items():
            if data["rescued"]:
                continue
                
            # Drift victim position with ocean current
            data["state"][:2] += ocean_drift + np.random.normal(0, 0.1, 2)
            tp = data["state"][:2]
            dist = np.linalg.norm(tp - uav)

            # Altitude factor: active branch = descended UAV -> tighter cov
            has_active_branch = (mode in ["aes_rarr", "static_R"] and data["active_branch"])
            alt_f = 0.05 if has_active_branch else 1.0 # improved covariance reduction
            sc = np.eye(2) * ((0.5 + 0.008 * dist) ** 2) * alt_f
            meas = tp + np.random.multivariate_normal([0, 0], sc)

            # --- Classification Phase (Real CNN vs Proxy) ---
            crop_img = get_class_crop_sample(data["class"])
            img_work = None
            if crop_img is not None:
                try:
                    img_work = crop_img.copy()
                    if data["occluded"]:
                        # Occlusion: Blur image crop heavily
                        img_work = img_work.filter(Image.ImageFilter.GaussianBlur(12.0))
                    if dist > 25.0:
                        # Long distance: downsample
                        img_work = img_work.resize((8, 8)).resize((64, 64))
                except Exception as e:
                    print(f"[Warning] Failed to apply crop effects: {e}")

            if cnn_model is not None and img_work is not None:
                try:
                    # Preprocess and forward pass
                    t = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_t = t(img_work).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        evidence = cnn_model(img_t).cpu().numpy()[0]
                        
                    alpha = evidence + 1.0
                    S_sum = np.sum(alpha)
                    beliefs_raw = evidence / S_sum
                    u_val = 5.0 / S_sum
                    probs_raw = alpha / S_sum
                    
                    # Map 5 classes to expected risk
                    # 0=swimmer, 1=floater, 2=boat, 3=life jacket, 4=buoy
                    risk_weights = np.array([1.0, 0.4, 0.2, 0.1, 0.05])
                    er = float(np.sum(probs_raw * risk_weights))
                    
                    # For metrics/belief reporting, we contract beliefs to 4 states (matching simulator format)
                    # 0: Drowning, 1: Floating, 2: Swimming, 3: Life jacket
                    beliefs = np.array([beliefs_raw[0] * 0.7, beliefs_raw[1], beliefs_raw[0] * 0.3 + beliefs_raw[2], beliefs_raw[3]])
                    beliefs /= (np.sum(beliefs) + 1e-6)
                    u = float(u_val)
                except Exception as e:
                    # Fallback to proxy
                    beliefs, u, probs = clf.estimate(data["class"], dist, data["occluded"])
                    er = float(np.sum(probs * clf.risk_weights))
            else:
                beliefs, u, probs = clf.estimate(data["class"], dist, data["occluded"])
                er = float(np.sum(probs * clf.risk_weights))

            vic_metrics[vid] = {
                "beliefs": beliefs,
                "u": u,
                "er": er,
                "score": 0.0,
                "crop_img": img_work
            }

            # Kalman update — dynamic R only for aes_rarr and no_branch
            g_use = gamma_r if mode in ["aes_rarr", "no_branch"] else 0.0
            u_use = u       if mode in ["aes_rarr", "no_branch"] else 0.0
            sts[vid] = trs[vid].update(
                trs[vid].predict(sts[vid], ocean_drift), meas, sc, u_use, g_use)

            # Active sensing branch trigger
            if mode in ["aes_rarr", "static_R"]:
                if best_last == vid and dist <= 30.0 and u > tau_unc and er > 0.25:
                    data["active_branch"] = True   # flag: descend next step
                    data["occluded"] = False       # descent clears occlusion!
                    branch_step = True
                    total_descent_attempts += 1
                    # Map to SeaDronesSee classes: class 0 is Drowning, 1 is Floating
                    if data["class"] in [0, 1]:  
                        true_positive_descents += 1
                else:
                    data["active_branch"] = False
                    if dist < 12.0:                # proximity clears occlusion
                        data["occluded"] = False
            else:
                data["active_branch"] = False

            # Priority score computation (Greedy scoring fallback)
            tt = dist / uav_speed
            if mode in ["aes_rarr", "static_R"] and data["active_branch"]:
                tt += descent_latency

            decay = decay_rates[data["class"]]
            if mode in ["aes_rarr", "no_branch", "static_R"]:
                # Risk-UCB
                p_raw = er + lambda_ra * u * (1.0 - er)
                score = (p_raw * math.exp(decay * tt)) / (dist + 1.0)
            elif mode == "deterministic":
                score = (er * math.exp(decay * tt)) / (dist + 1.0)
            else:
                score = 1.0 / (dist + 1.0)

            pris[vid] = score
            vic_metrics[vid]["score"] = score
            tpos[vid] = sts[vid][:2]

        if branch_step:
            branches += 1
            cumulative_time += descent_latency
        if not pris:
            break

        # Action Selection (Rollout vs Greedy)
        unrescued = {vid for vid, d in vics.items() if not d["rescued"]}
        if mode == "aes_rarr" and len(unrescued) > 0:
            best = select_next_target_lookahead(uav, unrescued, vics, uav_speed, decay_rates, descent_latency, depth=3)
        else:
            best = max(pris, key=pris.get)
            
        best_last = best
        d_dir = tpos[best] - uav
        d_norm = np.linalg.norm(d_dir)
        uav = tpos[best].copy() if d_norm <= uav_speed else uav + (d_dir / d_norm) * uav_speed

        # Rescue check
        for vid in list(pris.keys()):
            if np.linalg.norm(tpos[vid] - uav) <= rescue_dist and not vics[vid]["rescued"]:
                vics[vid]["rescued"] = True
                vics[vid]["rescue_time"] = cumulative_time

        if return_history:
            step_vic_data = {}
            for vid, data in vics.items():
                step_vic_data[vid] = {
                    "name": data["name"],
                    "true_pos": data["state"][:2].copy(),
                    "est_pos": sts[vid][:2].copy() if vid in sts else data["state"][:2].copy(),
                    "rescued": data["rescued"],
                    "rescue_time": data["rescue_time"],
                    "active_branch": data["active_branch"],
                    "occluded": data["occluded"],
                    "class": data["class"],
                    "beliefs": [float(b) for b in vic_metrics[vid]["beliefs"]],
                    "u": float(vic_metrics[vid]["u"]),
                    "er": float(vic_metrics[vid]["er"]),
                    "score": float(vic_metrics[vid]["score"]),
                    "crop_img": vic_metrics[vid].get("crop_img")
                }
            history.append({
                "step": step,
                "uav": uav.copy(),
                "uav_altitude": 20.0 if (mode in ["aes_rarr", "static_R"] and any(v["active_branch"] for v in vics.values())) else 50.0,
                "victims": step_vic_data,
                "priorities": pris.copy(),
                "best_target": best if 'best' in locals() else None,
                "cumulative_time": cumulative_time,
                "branches": branches
            })

    # Compute TTR and VSR per victim
    precision = true_positive_descents / total_descent_attempts if total_descent_attempts > 0 else 0.0
    results = []
    for vid, data in vics.items():
        mu = decay_rates[data["class"]]
        ttr = data["rescue_time"] if data["rescued"] else cumulative_time
        # VSR = survival prob at rescue (0.5x penalty if never rescued)
        vsr = math.exp(-mu * ttr) * (1.0 if data["rescued"] else 0.5)
        results.append({
            "Victim":        data["name"],
            "Mode":          mode,
            "TTR (steps)":   round(ttr, 2) if data["rescued"] else f">{round(cumulative_time, 2)}",
            "VSR":           round(vsr, 4),
            "Rescued":       "Yes" if data["rescued"] else "No",
            "TrueClass":     data["class"],
            "branch_precision": round(precision, 4)
        })
    if return_history:
        return results, branches, history
    return results, branches
