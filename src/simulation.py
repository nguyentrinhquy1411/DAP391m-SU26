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
                   victim_init=VICTIM_INIT_DEFAULT):
    """
    Runs target tracking and UAV search & rescue routing simulation under three modes:
    'aes_rarr' | 'deterministic' | 'distance_router'
    """
    clf = EvidentialClassifierSimulator()
    uav = np.array([0.0, 0.0])
    
    # Initialize victims
    vics = {vid: {
        "state":         np.array([d["pos"][0], d["pos"][1], 0.0, 0.0]),
        "class":         d["class"],
        "occluded":      d["occluded"],
        "name":          d["name"],
        "rescued":       False,
        "rescue_step":   None,
        "active_branch": False,
    } for vid, d in victim_init.items()}
    
    trs = {vid: UncertaintyKalmanFilter() for vid in vics}
    sts = {vid: vics[vid]["state"].copy() for vid in vics}
    branches = 0

    for step in range(1, sim_steps + 1):
        branch_step = False
        pris = {}
        tpos = {}

        for vid, data in vics.items():
            if data["rescued"]:
                continue
                
            # Drift victim position with ocean current
            data["state"][:2] += ocean_drift + np.random.normal(0, 0.1, 2)
            tp = data["state"][:2]
            dist = np.linalg.norm(tp - uav)

            # Altitude factor: active branch = descended UAV -> tighter cov
            alt_f = 0.25 if (mode == "aes_rarr" and data["active_branch"]) else 1.0
            sc = np.eye(2) * ((0.5 + 0.008 * dist) ** 2) * alt_f
            meas = tp + np.random.multivariate_normal([0, 0], sc)

            beliefs, u, probs = clf.estimate(data["class"], dist, data["occluded"])
            er = float(np.sum(probs * clf.risk_weights))

            # Kalman update — dynamic R only for AES-RARR
            g_use = gamma_r if mode == "aes_rarr" else 0.0
            u_use = u       if mode == "aes_rarr" else 0.0
            sts[vid] = trs[vid].update(
                trs[vid].predict(sts[vid], ocean_drift), meas, sc, u_use, g_use)

            # Active sensing branch trigger
            if mode == "aes_rarr":
                if u > tau_unc and er > 0.3:
                    data["active_branch"] = True   # flag: descend next step
                    branch_step = True
                else:
                    data["active_branch"] = False
                    if dist < 12.0:                # proximity clears occlusion
                        data["occluded"] = False

            # Priority score computation
            tt = dist / uav_speed
            decay = decay_rates[data["class"]]
            if mode == "aes_rarr":
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
            tpos[vid] = sts[vid][:2]

        if branch_step:
            branches += 1
        if not pris:
            break

        best = max(pris, key=pris.get)
        d_dir = tpos[best] - uav
        d_norm = np.linalg.norm(d_dir)
        uav = tpos[best].copy() if d_norm <= uav_speed else uav + (d_dir / d_norm) * uav_speed

        # Rescue check
        for vid in list(pris.keys()):
            if np.linalg.norm(tpos[vid] - uav) <= rescue_dist and not vics[vid]["rescued"]:
                vics[vid]["rescued"] = True
                vics[vid]["rescue_step"] = step

    # Compute TTR and VSR per victim
    results = []
    for vid, data in vics.items():
        mu = decay_rates[data["class"]]
        ttr = data["rescue_step"] if data["rescued"] else sim_steps
        # VSR = survival prob at rescue (0.5x penalty if never rescued)
        vsr = math.exp(-mu * ttr) * (1.0 if data["rescued"] else 0.5)
        results.append({
            "Victim":        data["name"],
            "Mode":          mode,
            "TTR (steps)":   ttr if data["rescued"] else f">{sim_steps}",
            "VSR":           round(vsr, 4),
            "Rescued":       "Yes" if data["rescued"] else "No",
        })
    return results, branches
