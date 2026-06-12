"""
Systematic parameter sensitivity analysis for AES-RARR framework.
Generates 1D sweeps and 2D heatmaps for key parameters.
"""
import numpy as np
import os
from src.simulation import run_simulation

def generate_random_pos():
    while True:
        pos = np.random.uniform(-70.0, 70.0, 2)
        if np.linalg.norm(pos) > 15.0:
            return pos

def quick_experiment(num_victims=3, occlusion_prob=0.30, num_trials=50,
                     tau_unc=0.55, lambda_ra=0.80, gamma_r=2.0,
                     num_distractors=2, descent_latency=1.0):
    """Run a quick Monte Carlo experiment with given parameters and return summary metrics."""
    modes = ["aes_rarr", "distance_router"]
    results = {m: {"worst_vsr": [], "cfr": 0} for m in modes}
    
    for trial in range(num_trials):
        victim_init = {}
        victim_init[1] = {
            "pos": generate_random_pos(), "class": 0,
            "occluded": (np.random.rand() < occlusion_prob),
            "name": "Victim 1 (Drowning)", "is_distractor": False
        }
        for idx in range(2, num_victims + 1):
            cls = np.random.choice([0, 1, 2, 3])
            cls_name = ["Drowning", "Floating", "Swimming", "PFD Floater"][cls]
            victim_init[idx] = {
                "pos": generate_random_pos(), "class": cls,
                "occluded": (np.random.rand() < occlusion_prob),
                "name": f"Victim {idx} ({cls_name})", "is_distractor": False
            }
        for d_idx in range(num_distractors):
            did = num_victims + d_idx + 1
            victim_init[did] = {
                "pos": generate_random_pos(), "class": np.random.choice([0, 1, 2, 3]),
                "occluded": (np.random.rand() < 0.5),
                "name": f"Distractor {d_idx+1} (Debris)", "is_distractor": True
            }
        
        trial_seed = np.random.randint(1000000)
        for mode in modes:
            np.random.seed(trial_seed)
            res_list, _ = run_simulation(
                mode=mode, victim_init=victim_init,
                tau_unc=tau_unc, lambda_ra=lambda_ra, gamma_r=gamma_r,
                descent_latency=descent_latency, sim_steps=30
            )
            real_res = [r for r in res_list if "Distractor" not in r["Victim"]]
            vsrs = [r["VSR"] for r in real_res]
            worst = min(vsrs) if vsrs else 0.0
            results[mode]["worst_vsr"].append(worst)
            if worst < 0.35:
                results[mode]["cfr"] += 1
    
    summary = {}
    for m in modes:
        summary[m] = {
            "worst_vsr_mean": np.mean(results[m]["worst_vsr"]),
            "worst_vsr_std": np.std(results[m]["worst_vsr"]),
            "cfr": results[m]["cfr"] / num_trials,
        }
    return summary


def sweep_1d(param_name, values, default_params, num_trials=50):
    """Run 1D parameter sweep."""
    print(f"\n{'='*60}")
    print(f"1D SWEEP: {param_name}")
    print(f"{'='*60}")
    results = []
    for val in values:
        params = default_params.copy()
        params[param_name] = val
        summary = quick_experiment(num_trials=num_trials, **params)
        aes = summary["aes_rarr"]
        dist = summary["distance_router"]
        delta = aes["worst_vsr_mean"] - dist["worst_vsr_mean"]
        results.append({
            "value": val,
            "aes_worst_vsr": aes["worst_vsr_mean"],
            "aes_cfr": aes["cfr"],
            "dist_worst_vsr": dist["worst_vsr_mean"],
            "delta_worst_vsr": delta,
        })
        print(f"  {param_name}={val:.2f} | AES Worst VSR={aes['worst_vsr_mean']:.3f} (CFR={aes['cfr']:.2f}) | "
              f"Dist Worst VSR={dist['worst_vsr_mean']:.3f} | Delta={delta:+.3f}")
    return results


def sweep_2d(param1_name, param1_values, param2_name, param2_values, 
             default_params, num_trials=30):
    """Run 2D parameter sweep (heatmap data)."""
    print(f"\n{'='*60}")
    print(f"2D SWEEP: {param1_name} x {param2_name}")
    print(f"{'='*60}")
    grid = np.zeros((len(param1_values), len(param2_values)))
    
    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            params = default_params.copy()
            params[param1_name] = v1
            params[param2_name] = v2
            summary = quick_experiment(num_trials=num_trials, **params)
            grid[i, j] = summary["aes_rarr"]["worst_vsr_mean"]
            print(f"  ({param1_name}={v1:.2f}, {param2_name}={v2:.2f}) -> Worst VSR = {grid[i, j]:.3f}")
    
    return grid


def sweep_decay_rates(rate_scales, default_params, num_trials=50):
    """Sweep survival decay rate scaling factors."""
    print(f"\n{'='*60}")
    print(f"SURVIVAL DECAY RATE SENSITIVITY")
    print(f"{'='*60}")
    from src.simulation import DECAY_RATES_DEFAULT
    results = []
    for scale in rate_scales:
        scaled_rates = {k: v * scale for k, v in DECAY_RATES_DEFAULT.items()}
        params = default_params.copy()
        # We need to pass decay_rates differently
        summary_aes = []
        summary_dist = []
        
        for trial in range(num_trials):
            victim_init = {}
            victim_init[1] = {
                "pos": generate_random_pos(), "class": 0,
                "occluded": (np.random.rand() < 0.30),
                "name": "V1 (Drowning)", "is_distractor": False
            }
            for idx in range(2, 4):
                cls = np.random.choice([0, 1, 2, 3])
                victim_init[idx] = {
                    "pos": generate_random_pos(), "class": cls,
                    "occluded": (np.random.rand() < 0.30),
                    "name": f"V{idx}", "is_distractor": False
                }
            seed = np.random.randint(1000000)
            
            np.random.seed(seed)
            res_aes, _ = run_simulation(mode="aes_rarr", victim_init=victim_init,
                                        decay_rates=scaled_rates, sim_steps=30)
            np.random.seed(seed)
            res_dist, _ = run_simulation(mode="distance_router", victim_init=victim_init,
                                         decay_rates=scaled_rates, sim_steps=30)
            
            vsrs_aes = [r["VSR"] for r in res_aes]
            vsrs_dist = [r["VSR"] for r in res_dist]
            summary_aes.append(min(vsrs_aes))
            summary_dist.append(min(vsrs_dist))
        
        results.append({
            "scale": scale,
            "aes_worst_vsr": np.mean(summary_aes),
            "dist_worst_vsr": np.mean(summary_dist),
            "delta": np.mean(summary_aes) - np.mean(summary_dist),
        })
        print(f"  Decay scale={scale:.1f}x | AES={np.mean(summary_aes):.3f} | Dist={np.mean(summary_dist):.3f} | Delta={results[-1]['delta']:+.3f}")
    
    return results


def main():
    np.random.seed(42)
    
    default_params = {
        "num_victims": 3,
        "occlusion_prob": 0.30,
        "num_distractors": 2,
        "tau_unc": 0.55,
        "lambda_ra": 0.80,
        "gamma_r": 2.0,
        "descent_latency": 1.0,
    }
    
    # 1D Sweeps
    tau_unc_results = sweep_1d("tau_unc", [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9], default_params)
    lambda_results = sweep_1d("lambda_ra", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], default_params)
    gamma_results = sweep_1d("gamma_r", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], default_params)
    descent_results = sweep_1d("descent_latency", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], default_params)
    
    # 2D Sweep: tau_unc x tau_risk (tau_risk is implicit in the er > 0.25 threshold)
    # We'll use occlusion_prob as a proxy for environmental difficulty
    tau_unc_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    occ_vals = [0.1, 0.2, 0.3, 0.5, 0.7]
    grid = sweep_2d("tau_unc", tau_unc_vals, "occlusion_prob", occ_vals, default_params, num_trials=30)
    
    # Decay rate sensitivity
    decay_results = sweep_decay_rates([0.5, 0.75, 1.0, 1.5, 2.0], default_params)
    
    # Print LaTeX-ready summary
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS SUMMARY (LaTeX-ready)")
    print(f"{'='*60}")
    
    print("\n% tau_unc sweep:")
    for r in tau_unc_results:
        print(f"% tau_unc={r['value']:.2f}: AES Worst VSR={r['aes_worst_vsr']:.3f}, CFR={r['aes_cfr']:.2f}, Delta={r['delta_worst_vsr']:+.3f}")
    
    print("\n% lambda sweep:")
    for r in lambda_results:
        print(f"% lambda={r['value']:.2f}: AES Worst VSR={r['aes_worst_vsr']:.3f}, CFR={r['aes_cfr']:.2f}")
    
    print("\n% gamma sweep:")
    for r in gamma_results:
        print(f"% gamma={r['value']:.2f}: AES Worst VSR={r['aes_worst_vsr']:.3f}, CFR={r['aes_cfr']:.2f}")
    
    print("\n% descent_latency sweep:")
    for r in descent_results:
        print(f"% descent_latency={r['value']:.2f}: AES Worst VSR={r['aes_worst_vsr']:.3f}, CFR={r['aes_cfr']:.2f}")
    
    print("\n% 2D heatmap (tau_unc x occlusion_prob) - Worst VSR values:")
    print(f"% tau_unc values: {tau_unc_vals}")
    print(f"% occlusion_prob values: {occ_vals}")
    for i, t in enumerate(tau_unc_vals):
        row_str = " & ".join([f"{grid[i,j]:.3f}" for j in range(len(occ_vals))])
        print(f"% tau_unc={t:.1f}: {row_str}")
    
    print("\n% Decay rate sensitivity:")
    for r in decay_results:
        print(f"% scale={r['scale']:.1f}x: AES={r['aes_worst_vsr']:.3f}, Dist={r['dist_worst_vsr']:.3f}, Delta={r['delta']:+.3f}")


if __name__ == "__main__":
    main()
