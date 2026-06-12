"""
Qualitative analysis: generates trajectory plots, uncertainty timelines,
and case comparisons for representative Monte Carlo trials.
"""
import numpy as np
import os
from src.simulation import run_simulation

def generate_random_pos():
    while True:
        pos = np.random.uniform(-70.0, 70.0, 2)
        if np.linalg.norm(pos) > 15.0:
            return pos


def run_single_trial_with_history(mode, victim_init, seed):
    """Run a single trial and return full step-by-step history."""
    np.random.seed(seed)
    results, branches, history = run_simulation(
        mode=mode, victim_init=victim_init,
        descent_latency=1.0, sim_steps=30,
        return_history=True
    )
    return results, branches, history


def plot_trajectory(history_aes, history_dist, victim_init, save_path="paper/figures/trajectory_comparison.png"):
    """Plot UAV trajectory comparison between AES-RARR and Distance Router."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Warning] matplotlib not available. Skipping trajectory plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    titles = ["AES-RARR (Active Sensing)", "Distance Router (Baseline)"]
    histories = [history_aes, history_dist]
    
    class_colors = {0: "#e74c3c", 1: "#f39c12", 2: "#3498db", 3: "#2ecc71"}
    class_names = {0: "Drowning", 1: "Floating", 2: "Swimming", 3: "PFD"}
    
    for ax, hist, title in zip(axes, histories, titles):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f0f4f8')
        
        # Plot UAV trajectory
        uav_xs = [h["uav"][0] for h in hist]
        uav_ys = [h["uav"][1] for h in hist]
        ax.plot(uav_xs, uav_ys, 'k-', alpha=0.4, linewidth=1)
        ax.plot(uav_xs[0], uav_ys[0], 'ks', markersize=8, label="UAV Start")
        ax.plot(uav_xs[-1], uav_ys[-1], 'k^', markersize=8, label="UAV End")
        
        # Plot victim positions
        for vid, vinit in victim_init.items():
            is_dist = vinit.get("is_distractor", False)
            cls = vinit["class"]
            pos = vinit["pos"]
            
            if is_dist:
                ax.plot(pos[0], pos[1], 'x', color='gray', markersize=10, markeredgewidth=2)
                ax.annotate(f"D{vid}", (pos[0]+1, pos[1]+1), fontsize=7, color='gray')
            else:
                color = class_colors.get(cls, "black")
                ax.plot(pos[0], pos[1], 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
                ax.annotate(f"V{vid}", (pos[0]+1, pos[1]+1), fontsize=8, fontweight='bold')
        
        # Mark branch descent events (altitude changes)
        for step_data in hist:
            if step_data.get("uav_altitude", 50) < 30:
                ax.plot(step_data["uav"][0], step_data["uav"][1], 
                       'v', color='purple', markersize=8, alpha=0.7)
        
        # Mark rescue events
        last_step = hist[-1]
        for vid, vdata in last_step["victims"].items():
            if vdata["rescued"]:
                ax.plot(vdata["true_pos"][0], vdata["true_pos"][1], 
                       '*', color='gold', markersize=15, markeredgecolor='black', markeredgewidth=0.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=class_colors[0], label='Drowning'),
        mpatches.Patch(color=class_colors[1], label='Floating'),
        mpatches.Patch(color=class_colors[2], label='Swimming'),
        mpatches.Patch(color=class_colors[3], label='PFD'),
        plt.Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=8, label='Distractor'),
        plt.Line2D([0], [0], marker='v', color='purple', linestyle='None', markersize=8, label='Descent'),
        plt.Line2D([0], [0], marker='*', color='gold', linestyle='None', markersize=12, label='Rescued'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=7, fontsize=9, 
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"  Saved trajectory plot to: {save_path}")
    plt.close()


def plot_uncertainty_timeline(history, victim_init, save_path="paper/figures/uncertainty_timeline.png"):
    """Plot epistemic uncertainty u(t) over time for each victim."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available. Skipping uncertainty timeline plot.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
    
    class_colors = {0: "#e74c3c", 1: "#f39c12", 2: "#3498db", 3: "#2ecc71"}
    
    # Top: Epistemic uncertainty u(t)
    ax = axes[0]
    ax.set_title("Epistemic Uncertainty Over Time", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Epistemic Uncertainty u(t)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.55, color='red', linestyle='--', alpha=0.5, label=r'$\tau_{unc}=0.55$')
    ax.grid(True, alpha=0.3)
    
    real_vids = [vid for vid, v in victim_init.items() if not v.get("is_distractor", False)]
    
    for vid in real_vids:
        steps = []
        u_vals = []
        for h in history:
            if vid in h["victims"]:
                steps.append(h["step"])
                u_vals.append(h["victims"][vid]["u"])
        cls = victim_init[vid]["class"]
        color = class_colors.get(cls, "black")
        name = victim_init[vid]["name"]
        ax.plot(steps, u_vals, '-o', color=color, markersize=3, label=f"{name}", linewidth=1.5)
    
    ax.legend(fontsize=8, loc='upper right')
    
    # Bottom: Expected risk E[R](t)
    ax2 = axes[1]
    ax2.set_title("Expected Risk Over Time", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Expected Risk E[R](t)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.30, color='orange', linestyle='--', alpha=0.5, label=r'$\tau_{risk}=0.30$')
    ax2.grid(True, alpha=0.3)
    
    for vid in real_vids:
        steps = []
        er_vals = []
        for h in history:
            if vid in h["victims"]:
                steps.append(h["step"])
                er_vals.append(h["victims"][vid]["er"])
        cls = victim_init[vid]["class"]
        color = class_colors.get(cls, "black")
        name = victim_init[vid]["name"]
        ax2.plot(steps, er_vals, '-s', color=color, markersize=3, label=f"{name}", linewidth=1.5)
    
    ax2.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"  Saved uncertainty timeline to: {save_path}")
    plt.close()


def main():
    np.random.seed(123)
    
    print("="*60)
    print("QUALITATIVE ANALYSIS")
    print("="*60)
    
    # Create a scenario with 3 victims + 2 distractors
    victim_init = {
        1: {"pos": np.array([55.0, 40.0]), "class": 0, "occluded": True,
            "name": "V1 (Drowning, Occluded)", "is_distractor": False},
        2: {"pos": np.array([25.0, -18.0]), "class": 2, "occluded": False,
            "name": "V2 (Swimming, Clear)", "is_distractor": False},
        3: {"pos": np.array([-35.0, 28.0]), "class": 1, "occluded": False,
            "name": "V3 (Floating)", "is_distractor": False},
        4: {"pos": np.array([15.0, 45.0]), "class": 1, "occluded": True,
            "name": "D1 (Buoy)", "is_distractor": True},
        5: {"pos": np.array([-20.0, -30.0]), "class": 2, "occluded": True,
            "name": "D2 (Debris)", "is_distractor": True},
    }
    
    seed = 42
    
    # Run both modes with full history
    print("\nRunning AES-RARR with full history...")
    res_aes, br_aes, hist_aes = run_single_trial_with_history("aes_rarr", victim_init, seed)
    
    print("Running Distance Router with full history...")
    res_dist, br_dist, hist_dist = run_single_trial_with_history("distance_router", victim_init, seed)
    
    # Print rescue comparison
    print(f"\n{'='*60}")
    print("RESCUE COMPARISON (Single Trial)")
    print(f"{'='*60}")
    print(f"\n{'Victim':<30} {'AES-RARR VSR':<15} {'Dist Router VSR':<15}")
    print("-" * 60)
    for r_aes, r_dist in zip(res_aes, res_dist):
        name = r_aes["Victim"]
        if "Distractor" in name:
            continue
        print(f"{name:<30} {r_aes['VSR']:<15.4f} {r_dist['VSR']:<15.4f}")
    
    print(f"\nAES-RARR branches triggered: {br_aes}")
    
    # Generate plots
    print("\nGenerating trajectory comparison plot...")
    plot_trajectory(hist_aes, hist_dist, victim_init)
    
    print("Generating uncertainty timeline plot...")
    plot_uncertainty_timeline(hist_aes, victim_init)
    
    print("\nDone. Figures saved to paper/figures/")


if __name__ == "__main__":
    main()
