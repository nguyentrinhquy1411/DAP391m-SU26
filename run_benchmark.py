"""
Runtime and computational cost analysis for AES-RARR components.
Benchmarks inference latency of EDL models, Kalman filter, and priority computation.
Compares single-pass EDL vs MC Dropout vs Deep Ensemble approaches.
"""
import time
import numpy as np
import os

def benchmark_numpy_operations():
    """Benchmark pure NumPy operations (Kalman filter, priority computation)."""
    from src.tracker import UncertaintyKalmanFilter
    from src.simulator import EvidentialClassifierSimulator
    
    kf = UncertaintyKalmanFilter()
    clf = EvidentialClassifierSimulator()
    
    # Warm up
    state = np.array([10.0, 20.0, 0.0, 0.0])
    drift = np.array([0.1, -0.05])
    
    # Benchmark Kalman Filter predict+update
    n_iters = 10000
    start = time.perf_counter()
    for _ in range(n_iters):
        state_pred = kf.predict(state, drift)
        meas = state_pred[:2] + np.random.normal(0, 0.5, 2)
        sc = np.eye(2) * 0.25
        state = kf.update(state_pred, meas, sc, 0.5, 2.0)
    kf_time = (time.perf_counter() - start) / n_iters * 1000  # ms
    
    # Benchmark EDL Proxy Classifier
    start = time.perf_counter()
    for _ in range(n_iters):
        beliefs, u, probs = clf.estimate(0, 30.0, occluded=True)
    proxy_time = (time.perf_counter() - start) / n_iters * 1000
    
    # Benchmark Priority Computation
    import math
    start = time.perf_counter()
    for _ in range(n_iters):
        risk_weights = np.array([1.0, 0.4, 0.2, 0.1])
        er = float(np.sum(probs * risk_weights))
        p_raw = er + 0.8 * u * (1.0 - er)
        dist = 35.0
        tt = dist / 10.0
        score = (p_raw * math.exp(0.08 * tt)) / (dist + 1.0)
    priority_time = (time.perf_counter() - start) / n_iters * 1000
    
    return {
        "Kalman Filter (predict+update)": kf_time,
        "EDL Proxy Classifier": proxy_time,
        "Priority Computation": priority_time,
    }


def benchmark_pytorch_models():
    """Benchmark PyTorch model inference latency."""
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
    except ImportError:
        print("[Warning] PyTorch not available. Skipping PyTorch benchmarks.")
        return {}
    
    from src.models import EDLClassifier, EvidentialCNNClassifier, SoftmaxClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # --- EDL MLP ---
    mlp = EDLClassifier(input_dim=7, num_classes=5).to(device)
    mlp.eval()
    x_mlp = torch.randn(1, 7).to(device)
    
    # Warm up
    for _ in range(100):
        with torch.no_grad():
            _ = mlp(x_mlp)
    
    n_iters = 5000
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = mlp(x_mlp)
    mlp_time = (time.perf_counter() - start) / n_iters * 1000
    results["EDL MLP (7-dim input)"] = mlp_time
    
    # --- EDL CNN (MobileNetV3-Small) ---
    cnn = EvidentialCNNClassifier(num_classes=5).to(device)
    cnn.eval()
    x_cnn = torch.randn(1, 3, 64, 64).to(device)
    
    for _ in range(50):
        with torch.no_grad():
            _ = cnn(x_cnn)
    
    n_iters_cnn = 500
    start = time.perf_counter()
    for _ in range(n_iters_cnn):
        with torch.no_grad():
            _ = cnn(x_cnn)
    cnn_time = (time.perf_counter() - start) / n_iters_cnn * 1000
    results["EDL CNN (MobileNetV3-Small, 64x64)"] = cnn_time
    
    # --- MC Dropout Simulation (10 forward passes) ---
    # Create a dropout-enabled MLP
    class MCDropoutMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(7, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 5)
            )
        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)
    
    mc_model = MCDropoutMLP().to(device)
    mc_model.train()  # Keep dropout active
    
    for mc_passes in [10, 50]:
        for _ in range(50):
            with torch.no_grad():
                _ = torch.stack([mc_model(x_mlp) for _ in range(mc_passes)])
        
        n_mc_iters = 500
        start = time.perf_counter()
        for _ in range(n_mc_iters):
            with torch.no_grad():
                preds = torch.stack([mc_model(x_mlp) for _ in range(mc_passes)])
                mean_pred = preds.mean(dim=0)
                epistemic_unc = preds.var(dim=0).sum()
        mc_time = (time.perf_counter() - start) / n_mc_iters * 1000
        results[f"MC Dropout MLP ({mc_passes} passes)"] = mc_time
    
    # --- Deep Ensemble Simulation (3 and 5 models) ---
    for n_models in [3, 5]:
        ensemble = [SoftmaxClassifier(input_dim=7, num_classes=5).to(device) for _ in range(n_models)]
        for m in ensemble:
            m.eval()
        
        for _ in range(50):
            with torch.no_grad():
                _ = torch.stack([m(x_mlp) for m in ensemble])
        
        n_ens_iters = 1000
        start = time.perf_counter()
        for _ in range(n_ens_iters):
            with torch.no_grad():
                preds = torch.stack([m(x_mlp) for m in ensemble])
                mean_pred = preds.mean(dim=0)
                epistemic_unc = preds.var(dim=0).sum()
        ens_time = (time.perf_counter() - start) / n_ens_iters * 1000
        results[f"Deep Ensemble MLP ({n_models} models)"] = ens_time
    
    # --- Model parameter counts ---
    mlp_params = sum(p.numel() for p in mlp.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())
    results["_mlp_params"] = mlp_params
    results["_cnn_params"] = cnn_params
    
    return results


def main():
    print("="*70)
    print("RUNTIME / COMPUTATIONAL COST ANALYSIS")
    print("="*70)
    
    # NumPy benchmarks
    print("\n--- NumPy-based Components ---")
    np_results = benchmark_numpy_operations()
    for name, time_ms in np_results.items():
        print(f"  {name:<40} {time_ms:.4f} ms")
    
    # PyTorch benchmarks
    print("\n--- PyTorch Model Inference ---")
    pt_results = benchmark_pytorch_models()
    
    if pt_results:
        import torch
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"  Device: {device_name}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        for name, val in pt_results.items():
            if name.startswith("_"):
                continue
            print(f"  {name:<45} {val:.4f} ms")
        
        if "_mlp_params" in pt_results:
            print(f"\n  EDL MLP parameters:  {pt_results['_mlp_params']:,}")
            print(f"  EDL CNN parameters:  {pt_results['_cnn_params']:,}")
    
    # Print total per-target per-step cost
    print("\n--- Per-Target Per-Step Total Latency ---")
    total_edl_mlp = np_results.get("Kalman Filter (predict+update)", 0) + \
                    pt_results.get("EDL MLP (7-dim input)", 0) + \
                    np_results.get("Priority Computation", 0)
    total_edl_cnn = np_results.get("Kalman Filter (predict+update)", 0) + \
                    pt_results.get("EDL CNN (MobileNetV3-Small, 64x64)", 0) + \
                    np_results.get("Priority Computation", 0)
    
    print(f"  EDL MLP pipeline:  {total_edl_mlp:.4f} ms/target")
    print(f"  EDL CNN pipeline:  {total_edl_cnn:.4f} ms/target")
    
    n_targets = 10
    print(f"\n  For N={n_targets} targets:")
    print(f"    EDL MLP: {total_edl_mlp * n_targets:.2f} ms/step ({1000 / (total_edl_mlp * n_targets):.0f} Hz)")
    print(f"    EDL CNN: {total_edl_cnn * n_targets:.2f} ms/step ({1000 / (total_edl_cnn * n_targets):.0f} Hz)")
    
    # LaTeX table output
    print(f"\n{'='*70}")
    print("% LATEX TABLE: Runtime Comparison")
    print(f"{'='*70}")
    print(r"""\begin{tabularx}{\textwidth}{YCC}
\toprule
\textbf{Method} & \textbf{Latency (ms)} & \textbf{Parameters} \\
\midrule""")
    
    all_results = {**np_results, **{k: v for k, v in pt_results.items() if not k.startswith("_")}}
    for name, val in all_results.items():
        params_str = ""
        if "MLP" in name and "MC" not in name and "Ensemble" not in name:
            params_str = f"{pt_results.get('_mlp_params', 'N/A'):,}"
        elif "CNN" in name:
            params_str = f"{pt_results.get('_cnn_params', 'N/A'):,}"
        else:
            params_str = "---"
        print(f"{name} & {val:.4f} & {params_str} \\\\")
    
    print(r"""\bottomrule
\end{tabularx}""")


if __name__ == "__main__":
    main()
