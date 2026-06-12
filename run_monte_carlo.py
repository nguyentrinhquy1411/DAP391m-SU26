import numpy as np
import math
import sys
from src.simulation import run_simulation

def generate_random_pos():
    while True:
        pos = np.random.uniform(-70.0, 70.0, 2)
        if np.linalg.norm(pos) > 15.0:
            return pos

def run_experiment(num_victims, occlusion_prob, num_trials=100):
    modes = ["aes_rarr", "no_branch", "static_R", "deterministic", "distance_router"]
    
    results = {mode: {
        "worst_vsr": [],
        "mean_vsr": [],
        "mean_ttr": [],
        "cfr": 0,
        "branch_precisions": [],
        "total_branches": []
    } for mode in modes}
    
    for trial in range(num_trials):
        # Generate shared victim initialization for this trial
        victim_init = {}
        # Make sure we have at least one drowning victim
        victim_init[1] = {
            "pos": generate_random_pos(),
            "class": 0,
            "occluded": (np.random.rand() < occlusion_prob),
            "name": "Victim 1 (Drowning)"
        }
        for idx in range(2, num_victims + 1):
            cls = np.random.choice([0, 1, 2, 3])
            cls_name = ["Drowning", "Floating", "Swimming", "PFD Floater"][cls]
            victim_init[idx] = {
                "pos": generate_random_pos(),
                "class": cls,
                "occluded": (np.random.rand() < occlusion_prob),
                "name": f"Victim {idx} ({cls_name})"
            }
            
        # Save random state to ensure identical environment updates (drift & noise) across modes in a trial
        trial_seed = np.random.randint(1000000)
        
        for mode in modes:
            np.random.seed(trial_seed)
            res_list, branches = run_simulation(
                mode=mode, 
                victim_init=victim_init, 
                descent_latency=1.0, 
                sim_steps=40 if num_victims > 5 else 30
            )
            
            vsrs = [r["VSR"] for r in res_list]
            ttrs = []
            for r in res_list:
                t_str = r["TTR (steps)"]
                if isinstance(t_str, str) and t_str.startswith(">"):
                    ttrs.append(float(t_str[1:]))
                else:
                    ttrs.append(float(t_str))
                    
            worst_vsr = min(vsrs)
            mean_vsr = np.mean(vsrs)
            mean_ttr = np.mean(ttrs)
            
            results[mode]["worst_vsr"].append(worst_vsr)
            results[mode]["mean_vsr"].append(mean_vsr)
            results[mode]["mean_ttr"].append(mean_ttr)
            results[mode]["total_branches"].append(branches)
            
            if worst_vsr < 0.35:
                results[mode]["cfr"] += 1
                
            if mode in ["aes_rarr", "static_R"]:
                results[mode]["branch_precisions"].append(res_list[0]["branch_precision"])
                
    summary = {}
    for mode in modes:
        w_vsr_m = np.mean(results[mode]["worst_vsr"])
        w_vsr_s = np.std(results[mode]["worst_vsr"])
        m_vsr_m = np.mean(results[mode]["mean_vsr"])
        m_vsr_s = np.std(results[mode]["mean_vsr"])
        m_ttr_m = np.mean(results[mode]["mean_ttr"])
        m_ttr_s = np.std(results[mode]["mean_ttr"])
        cfr_val = results[mode]["cfr"] / num_trials
        avg_branches = np.mean(results[mode]["total_branches"])
        
        if mode in ["aes_rarr", "static_R"] and len(results[mode]["branch_precisions"]) > 0:
            prec_m = np.mean(results[mode]["branch_precisions"])
            prec_s = np.std(results[mode]["branch_precisions"])
            prec_str = f"{prec_m:.2f} \u00b1 {prec_s:.2f}"
        else:
            prec_str = "n/a"
            
        summary[mode] = {
            "worst_vsr": (w_vsr_m, w_vsr_s),
            "mean_vsr": (m_vsr_m, m_vsr_s),
            "mean_ttr": (m_ttr_m, m_ttr_s),
            "cfr": cfr_val,
            "precision_str": prec_str,
            "branches": avg_branches
        }
    return summary

def main():
    np.random.seed(42)
    
    print("Running Experiment 1: N=3 victims, 30% occlusion (Main Result)...")
    exp1 = run_experiment(num_victims=3, occlusion_prob=0.30)
    
    print("\nRunning Experiment 2: N=3 victims, 70% occlusion (High Occlusion)...")
    exp2 = run_experiment(num_victims=3, occlusion_prob=0.70)
    
    print("\nRunning Experiment 3: N=10 victims, 30% occlusion (High Victim Density)...")
    exp3 = run_experiment(num_victims=10, occlusion_prob=0.30)
    
    print("\n" + "="*80)
    print("MONTE CARLO EXPERIMENTS SUMMARY")
    print("="*80)
    
    # Print Table 1 LaTeX code
    print("\n% TABLE 1 LATEX CODE:")
    print(r"""\begin{tabularx}{\textwidth}{p{2.8cm}CCCCC}
\toprule
\textbf{Method} & \makecell{\textbf{Worst}\\\textbf{VSR}} & \makecell{\textbf{Mean}\\\textbf{VSR}} & \makecell{\textbf{Mean}\\\textbf{TTR}} & \textbf{CFR} & \makecell{\textbf{Branch}\\\textbf{Precision}} \\
\midrule""")
    
    method_latex_map = {
        "aes_rarr": r"\aes{} full",
        "no_branch": "No active branch",
        "static_R": "Static measurement covariance",
        "deterministic": "Deterministic expected risk",
        "distance_router": "Distance router"
    }
    
    for mode in ["aes_rarr", "no_branch", "static_R", "deterministic", "distance_router"]:
        m = exp1[mode]
        worst_str = f"${m['worst_vsr'][0]:.2f}\\pm{m['worst_vsr'][1]:.2f}$"
        mean_vsr_str = f"${m['mean_vsr'][0]:.2f}\\pm{m['mean_vsr'][1]:.2f}$"
        mean_ttr_str = f"${m['mean_ttr'][0]:.2f}\\pm{m['mean_ttr'][1]:.2f}$"
        cfr_str = f"{m['cfr']:.2f}"
        prec_str = m['precision_str']
        if prec_str != "n/a":
            prec_str = f"${prec_str}$"
        
        # Bold best values (worst vsr, mean vsr, cfr, precision)
        # For exp1:
        if mode == "aes_rarr":
            worst_str = r"\textbf{" + worst_str + "}"
            cfr_str = r"\textbf{" + cfr_str + "}"
            prec_str = r"\textbf{" + prec_str + "}"
        elif mode == "deterministic":
            mean_vsr_str = r"\textbf{" + mean_vsr_str + "}"
            
        print(f"{method_latex_map[mode]:<30} & {worst_str} & {mean_vsr_str} & {mean_ttr_str} & {cfr_str} & {prec_str} \\\\")
        
    print(r"""\bottomrule
\end{tabularx}""")
    
    # Print Table 2 LaTeX code
    print("\n% TABLE 2 LATEX CODE:")
    print(r"""\begin{tabularx}{\textwidth}{lCCC}
\toprule
\textbf{Condition} & \textbf{Method} & \makecell{\textbf{Worst}\\\textbf{VSR}} & \textbf{CFR} \\
\midrule""")
    
    # 70% occlusion
    m_aes_70 = exp2["aes_rarr"]
    m_det_70 = exp2["deterministic"]
    print(f"70\\% initial occlusion & \\\\aes{{}} full & ${m_aes_70['worst_vsr'][0]:.2f}\\pm{m_aes_70['worst_vsr'][1]:.2f}$ & {m_aes_70['cfr']:.2f} \\\\")
    print(f"70\\% initial occlusion & Deterministic expected risk & ${m_det_70['worst_vsr'][0]:.2f}\\pm{m_det_70['worst_vsr'][1]:.2f}$ & {m_det_70['cfr']:.2f} \\\\")
    print(r"\midrule")
    
    # N=10 victims
    m_aes_10 = exp3["aes_rarr"]
    m_det_10 = exp3["deterministic"]
    print(f"$N=10$ victims & \\\\aes{{}} full & ${m_aes_10['worst_vsr'][0]:.2f}\\pm{m_aes_10['worst_vsr'][1]:.2f}$ & {m_aes_10['cfr']:.2f} \\\\")
    print(f"$N=10$ victims & Deterministic expected risk & ${m_det_10['worst_vsr'][0]:.2f}\\pm{m_det_10['worst_vsr'][1]:.2f}$ & {m_det_10['cfr']:.2f} \\\\")
    
    print(r"""\bottomrule
\end{tabularx}""")

if __name__ == "__main__":
    main()
