import numpy as np
import math
from .tracker import UncertaintyKalmanFilter
from .simulator import EvidentialClassifierSimulator

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
            alt_f = 0.25 if has_active_branch else 1.0
            sc = np.eye(2) * ((0.5 + 0.008 * dist) ** 2) * alt_f
            meas = tp + np.random.multivariate_normal([0, 0], sc)

            beliefs, u, probs = clf.estimate(data["class"], dist, data["occluded"])
            er = float(np.sum(probs * clf.risk_weights))
            vic_metrics[vid] = {
                "beliefs": beliefs,
                "u": u,
                "er": er,
                "score": 0.0
            }

            # Kalman update — dynamic R only for aes_rarr and no_branch
            g_use = gamma_r if mode in ["aes_rarr", "no_branch"] else 0.0
            u_use = u       if mode in ["aes_rarr", "no_branch"] else 0.0
            sts[vid] = trs[vid].update(
                trs[vid].predict(sts[vid], ocean_drift), meas, sc, u_use, g_use)

            # Active sensing branch trigger
            if mode in ["aes_rarr", "static_R"]:
                if best_last == vid and dist <= 20.0 and u > tau_unc and er > 0.3:
                    data["active_branch"] = True   # flag: descend next step
                    data["occluded"] = False       # descent clears occlusion!
                    branch_step = True
                    total_descent_attempts += 1
                    if data["class"] in [0, 1]:  # Drowning (0) or Floating (1)
                        true_positive_descents += 1
                else:
                    data["active_branch"] = False
                    if dist < 12.0:                # proximity clears occlusion
                        data["occluded"] = False
            else:
                data["active_branch"] = False

            # Priority score computation
            tt = dist / uav_speed
            if mode in ["aes_rarr", "static_R"] and data["active_branch"]:
                tt += descent_latency

            decay = decay_rates[data["class"]]
            if mode in ["aes_rarr", "no_branch", "static_R"]:
                # Risk-UCB: boosts uncertain high-risk victims early
                p_raw = er + lambda_ra * u * (1.0 - er)
                score = (p_raw * math.exp(decay * tt)) / (dist + 1.0)
            elif mode == "deterministic":
                # Expected-risk only, no UCB uncertainty boost
                score = (er * math.exp(decay * tt)) / (dist + 1.0)
            else:
                # Distance router: nearest-first, ignores risk entirely
                score = 1.0 / (dist + 1.0)

            pris[vid] = score
            vic_metrics[vid]["score"] = score
            tpos[vid] = sts[vid][:2]

        if branch_step:
            branches += 1
            cumulative_time += descent_latency
        if not pris:
            break

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
                    "score": float(vic_metrics[vid]["score"])
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
