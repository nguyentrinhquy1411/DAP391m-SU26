import numpy as np
import math
import sys
from scipy import stats as scipy_stats
from src.simulation import run_simulation

def generate_random_pos():
    while True:
        pos = np.random.uniform(-70.0, 70.0, 2)
        if np.linalg.norm(pos) > 15.0:
            return pos

def cohens_d(x, y):
    """Compute Cohen's d effect size for paired samples."""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

def run_experiment(num_victims, occlusion_prob, num_trials=500, 
                   num_distractors=0, sea_state="moderate"):
    modes = ["aes_rarr", "no_branch", "static_R", "deterministic", "distance_router"]
    
    results = {mode: {
        "worst_vsr": [],
        "mean_vsr": [],
        "mean_ttr": [],
        "cfr": 0,
        "branch_precisions": [],
        "total_branches": [],
        "distractors_filtered": [],
        "distractors_visited": [],
        "rpes": [],
    } for mode in modes}
    
    # Sea state affects occlusion dynamics
    sea_state_occlusion_boost = {"calm": -0.1, "moderate": 0.0, "rough": 0.15}
    occ_boost = sea_state_occlusion_boost.get(sea_state, 0.0)
    effective_occ = min(0.95, max(0.05, occlusion_prob + occ_boost))
    
    for trial in range(num_trials):
        # Generate shared victim initialization for this trial
        victim_init = {}
        # Make sure we have at least one drowning victim
        victim_init[1] = {
            "pos": generate_random_pos(),
            "class": 0,
            "occluded": (np.random.rand() < effective_occ),
            "name": "Victim 1 (Drowning)",
            "is_distractor": False
        }
        for idx in range(2, num_victims + 1):
            cls = np.random.choice([0, 1, 2, 3])
            cls_name = ["Drowning", "Floating", "Swimming", "PFD Floater"][cls]
            victim_init[idx] = {
                "pos": generate_random_pos(),
                "class": cls,
                "occluded": (np.random.rand() < effective_occ),
                "name": f"Victim {idx} ({cls_name})",
                "is_distractor": False
            }
        
        # Add distractors (buoys, debris) — these are NOT real victims
        for d_idx in range(num_distractors):
            did = num_victims + d_idx + 1
            d_type = np.random.choice(["Buoy", "Debris", "Wave Object"])
            victim_init[did] = {
                "pos": generate_random_pos(),
                "class": np.random.choice([0, 1, 2, 3]),  # Ambiguous appearance
                "occluded": (np.random.rand() < 0.5),       # Often partially visible
                "name": f"Distractor {d_idx+1} ({d_type})",
                "is_distractor": True
            }
            
        # Save random state to ensure identical environment updates across modes
        trial_seed = np.random.randint(1000000)
        
        for mode in modes:
            np.random.seed(trial_seed)
            res_list, branches = run_simulation(
                mode=mode, 
                victim_init=victim_init, 
                descent_latency=1.0, 
                sim_steps=40 if (num_victims + num_distractors) > 5 else 30,
                max_payload=num_victims
            )
            
            # Separate real victims from distractors in results
            real_results = [r for r in res_list if "Distractor" not in r["Victim"]]
            distractor_results = [r for r in res_list if "Distractor" in r["Victim"]]
            
            vsrs = [r["VSR"] for r in real_results]
            ttrs = []
            for r in real_results:
                t_str = r["TTR (steps)"]
                if isinstance(t_str, str) and t_str.startswith(">"):
                    ttrs.append(float(t_str[1:]))
                else:
                    ttrs.append(float(t_str))
                    
            worst_vsr = min(vsrs) if vsrs else 0.0
            mean_vsr = np.mean(vsrs) if vsrs else 0.0
            mean_ttr = np.mean(ttrs) if ttrs else 0.0
            
            # Count distractors that were visited (wasted time) vs filtered
            dist_visited = sum(1 for r in distractor_results if r["Rescued"] == "Yes")
            dist_filtered = len(distractor_results) - dist_visited
            
            # Compute Rescue Package Efficiency (RPE)
            real_rescued = sum(1 for r in real_results if r["Rescued"] == "Yes")
            total_dropped = real_rescued + dist_visited
            rpe = real_rescued / total_dropped if total_dropped > 0 else 1.0
            
            results[mode]["worst_vsr"].append(worst_vsr)
            results[mode]["mean_vsr"].append(mean_vsr)
            results[mode]["mean_ttr"].append(mean_ttr)
            results[mode]["total_branches"].append(branches)
            results[mode]["distractors_filtered"].append(dist_filtered)
            results[mode]["distractors_visited"].append(dist_visited)
            results[mode]["rpes"].append(rpe)
            
            if worst_vsr < 0.35:
                results[mode]["cfr"] += 1
                
            if mode in ["aes_rarr", "static_R"]:
                results[mode]["branch_precisions"].append(real_results[0]["branch_precision"] if real_results else 0.0)
                
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
            
        rpe_m = np.mean(results[mode]["rpes"])
        rpe_s = np.std(results[mode]["rpes"])
        
        summary[mode] = {
            "worst_vsr": (w_vsr_m, w_vsr_s),
            "mean_vsr": (m_vsr_m, m_vsr_s),
            "mean_ttr": (m_ttr_m, m_ttr_s),
            "cfr": cfr_val,
            "precision_str": prec_str,
            "branches": avg_branches,
            "rpe": (rpe_m, rpe_s),
            "raw_worst_vsr": results[mode]["worst_vsr"],
            "raw_mean_vsr": results[mode]["mean_vsr"],
            "raw_mean_ttr": results[mode]["mean_ttr"],
            "distractors_filtered_mean": np.mean(results[mode]["distractors_filtered"]),
            "distractors_visited_mean": np.mean(results[mode]["distractors_visited"]),
        }
    return summary

def compute_statistical_tests(summary, reference_mode="aes_rarr"):
    """Compute Wilcoxon signed-rank tests and Cohen's d between reference and all baselines."""
    ref_data = summary[reference_mode]
    test_results = {}
    
    for mode in summary:
        if mode == reference_mode:
            continue
        mode_data = summary[mode]
        tests = {}
        
        for metric in ["raw_worst_vsr", "raw_mean_vsr", "raw_mean_ttr"]:
            ref_vals = np.array(ref_data[metric])
            cmp_vals = np.array(mode_data[metric])
            
            # Wilcoxon signed-rank test (paired, non-parametric)
            try:
                stat, p_val = scipy_stats.wilcoxon(ref_vals, cmp_vals, alternative='two-sided')
            except ValueError:
                # All differences are zero
                stat, p_val = 0.0, 1.0
            
            d = cohens_d(ref_vals, cmp_vals)
            
            metric_name = metric.replace("raw_", "")
            tests[metric_name] = {
                "wilcoxon_stat": stat,
                "p_value": p_val,
                "cohens_d": d,
                "significant": p_val < 0.05
            }
        test_results[mode] = tests
    
    return test_results


def print_statistical_results(test_results, reference_mode="aes_rarr"):
    """Print statistical significance table."""
    print(f"\n{'='*90}")
    print(f"STATISTICAL SIGNIFICANCE TESTS (Reference: {reference_mode})")
    print(f"{'='*90}")
    print(f"{'Comparison':<35} {'Metric':<15} {'p-value':<12} {'Cohen d':<10} {'Sig?':<6}")
    print("-" * 90)
    
    for mode, tests in test_results.items():
        for metric, result in tests.items():
            sig_mark = "***" if result["p_value"] < 0.001 else \
                       "**" if result["p_value"] < 0.01 else \
                       "*" if result["p_value"] < 0.05 else "ns"
            print(f"vs {mode:<32} {metric:<15} {result['p_value']:<12.4f} {result['cohens_d']:<10.3f} {sig_mark:<6}")


def main():
    np.random.seed(42)
    
    # ============================================================
    # Experiment 1: Main result — N=3, 30% occlusion, 2 distractors
    # ============================================================
    print("Running Experiment 1: N=3 victims + 2 distractors, 30% occlusion...")
    exp1 = run_experiment(num_victims=3, occlusion_prob=0.30, num_distractors=2)
    stat1 = compute_statistical_tests(exp1)
    print_statistical_results(stat1)
    
    # ============================================================
    # Experiment 2: High occlusion — N=3, 70% occlusion, 2 distractors
    # ============================================================
    print("\nRunning Experiment 2: N=3, 70% occlusion, 2 distractors...")
    exp2 = run_experiment(num_victims=3, occlusion_prob=0.70, num_distractors=2)
    
    # ============================================================
    # Experiment 3: High density — N=5, 30% occlusion, 3 distractors
    # ============================================================
    print("\nRunning Experiment 3: N=5 victims + 3 distractors, 30% occlusion...")
    exp3 = run_experiment(num_victims=5, occlusion_prob=0.30, num_distractors=3)
    stat3 = compute_statistical_tests(exp3)
    print_statistical_results(stat3)
    
    # ============================================================
    # Experiment 4: Very high density — N=10, 30% occlusion
    # ============================================================
    print("\nRunning Experiment 4: N=10 victims, 30% occlusion...")
    exp4 = run_experiment(num_victims=10, occlusion_prob=0.30, num_distractors=0)
    
    # ============================================================
    # Experiment 5: N=1 victim (edge case), 50% occlusion
    # ============================================================
    print("\nRunning Experiment 5: N=1 victim, 50% occlusion...")
    exp5 = run_experiment(num_victims=1, occlusion_prob=0.50, num_distractors=2)
    
    # ============================================================
    # Experiment 6: N=8 victims, 30% occlusion, 4 distractors
    # ============================================================
    print("\nRunning Experiment 6: N=8 victims + 4 distractors, 30% occlusion...")
    exp6 = run_experiment(num_victims=8, occlusion_prob=0.30, num_distractors=4)
    
    print("\n" + "="*80)
    print("MONTE CARLO EXPERIMENTS SUMMARY")
    print("="*80)
    
    # Print Table 1 LaTeX code (Main results with distractors)
    print("\n% TABLE 1 LATEX CODE (N=3, 30% occ, 2 distractors):")
    method_latex_map = {
        "aes_rarr": r"\aes{} full",
        "no_branch": "No active branch",
        "static_R": "Static measurement covariance",
        "deterministic": "Deterministic expected risk",
        "distance_router": "Distance router"
    }
    
    print(r"""\begin{tabularx}{\textwidth}{p{2.8cm}CCCCCC}
\toprule
\textbf{Method} & \makecell{\textbf{Worst}\\\textbf{VSR}} & \makecell{\textbf{Mean}\\\textbf{VSR}} & \makecell{\textbf{Mean}\\\textbf{TTR}} & \textbf{CFR} & \textbf{RPE} & \makecell{\textbf{Branch}\\\textbf{Precision}} \\
\midrule""")
    
    for mode in ["aes_rarr", "no_branch", "static_R", "deterministic", "distance_router"]:
        m = exp1[mode]
        worst_str = f"${m['worst_vsr'][0]:.2f}\\pm{m['worst_vsr'][1]:.2f}$"
        mean_vsr_str = f"${m['mean_vsr'][0]:.2f}\\pm{m['mean_vsr'][1]:.2f}$"
        mean_ttr_str = f"${m['mean_ttr'][0]:.2f}\\pm{m['mean_ttr'][1]:.2f}$"
        cfr_str = f"{m['cfr']:.2f}"
        rpe_str = f"${m['rpe'][0]:.2f}\\pm{m['rpe'][1]:.2f}$"
        prec_str = m['precision_str']
        if prec_str != "n/a":
            prec_str = f"${prec_str}$"
        
        print(f"{method_latex_map[mode]:<30} & {worst_str} & {mean_vsr_str} & {mean_ttr_str} & {cfr_str} & {rpe_str} & {prec_str} \\\\")
        
    print(r"""\bottomrule
\end{tabularx}""")
    
    # Print statistical significance summary for Table 1
    print("\n% STATISTICAL SIGNIFICANCE (vs AES-RARR full, Worst VSR):")
    for mode, tests in stat1.items():
        t = tests["worst_vsr"]
        sig = "p<0.001" if t["p_value"] < 0.001 else \
              "p<0.01" if t["p_value"] < 0.01 else \
              "p<0.05" if t["p_value"] < 0.05 else "n.s."
        print(f"% vs {mode}: p={t['p_value']:.4f} ({sig}), Cohen's d={t['cohens_d']:.3f}")
    
    # Print Table 2 LaTeX code (Robustness: occlusion + density + distractors)
    print("\n% TABLE 2 LATEX CODE (Robustness):")
    print(r"""\begin{tabularx}{\textwidth}{lCCC}
\toprule
\textbf{Condition} & \textbf{Method} & \makecell{\textbf{Worst}\\\textbf{VSR}} & \textbf{CFR} \\
\midrule""")
    
    # 70% occlusion
    for mode_key, mode_label in [("aes_rarr", r"\\aes{} full"), ("deterministic", "Deterministic expected risk")]:
        m = exp2[mode_key]
        print(f"70\\% initial occlusion & {mode_label} & ${m['worst_vsr'][0]:.2f}\\pm{m['worst_vsr'][1]:.2f}$ & {m['cfr']:.2f} \\\\")
    print(r"\midrule")
    
    # N=5 + distractors
    for mode_key, mode_label in [("aes_rarr", r"\\aes{} full"), ("deterministic", "Deterministic expected risk")]:
        m = exp3[mode_key]
        print(f"$N=5$ + 3 distractors & {mode_label} & ${m['worst_vsr'][0]:.2f}\\pm{m['worst_vsr'][1]:.2f}$ & {m['cfr']:.2f} \\\\")
    print(r"\midrule")
    
    # N=10
    for mode_key, mode_label in [("aes_rarr", r"\\aes{} full"), ("deterministic", "Deterministic expected risk")]:
        m = exp4[mode_key]
        print(f"$N=10$ victims & {mode_label} & ${m['worst_vsr'][0]:.2f}\\pm{m['worst_vsr'][1]:.2f}$ & {m['cfr']:.2f} \\\\")
    print(r"\midrule")
    
    # N=8 + distractors
    for mode_key, mode_label in [("aes_rarr", r"\\aes{} full"), ("deterministic", "Deterministic expected risk")]:
        m = exp6[mode_key]
        print(f"$N=8$ + 4 distractors & {mode_label} & ${m['worst_vsr'][0]:.2f}\\pm{m['worst_vsr'][1]:.2f}$ & {m['cfr']:.2f} \\\\")
    
    print(r"""\bottomrule
\end{tabularx}""")
    
    # Distractor filtering comparison
    print("\n% DISTRACTOR FILTERING TABLE:")
    print(f"{'Method':<35} {'Distractors Filtered (avg)':<28} {'Distractors Visited (avg)':<28}")
    print("-" * 90)
    for mode in ["aes_rarr", "no_branch", "deterministic", "distance_router"]:
        m = exp1[mode]
        print(f"{mode:<35} {m['distractors_filtered_mean']:<28.2f} {m['distractors_visited_mean']:<28.2f}")


if __name__ == "__main__":
    main()
