import os
import json
import numpy as np

# Configurations
local_paths = [
    "./archive/compressed/annotations/instances_val.json",
    "./archive/annotations/instances_val.json",
    "./sds-dataset/annotations/instances_val.json",
    "/content/sds-dataset/annotations/instances_val.json"
]
DATASET_JSON = "/content/sds-dataset/annotations/instances_val.json"
for path in local_paths:
    if os.path.exists(path):
        DATASET_JSON = os.path.abspath(path)
        break

MODEL_WEIGHTS_PATH = "edl_model_weights.npz"

def calculate_ece(probs, labels, num_bins=10):
    """Calculates the Expected Calibration Error (ECE) for evaluation."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    
    # Get max predicted probabilities and corresponding class predictions
    pred_probs = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select indices of samples in the current bin
        in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Accuracy in bin
            bin_acc = np.mean(pred_labels[in_bin] == labels[in_bin])
            # Confidence in bin
            bin_conf = np.mean(pred_probs[in_bin])
            ece += prop_in_bin * np.abs(bin_acc - bin_conf)
            
    return ece


def digamma(x):
    """NumPy Digamma function approximation (in case scipy is not installed)."""
    try:
        from scipy.special import psi
        return psi(x)
    except ImportError:
        # Simple polynomial approximation for x > 0
        r = 0.0
        while x < 7:
            r -= 1.0 / x
            x += 1.0
        x -= 0.5
        xx = 1.0 / x
        xx2 = xx * xx
        return r + np.log(x) + (1./12. - (1./120. - 1./252. * xx2) * xx2) * xx2


def train_pytorch(json_path):
    """Standard PyTorch implementation of the Evidential Deep Learning Pipeline."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    # 1. Parse dataset annotations
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    
    # Categories: swimmer, floater, boat, life jacket, buoy
    categories = sorted(list({ann["category_id"] for ann in annotations}))
    num_classes = len(categories)
    cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(categories)}
    
    print(f"Dataset categories parsed: {num_classes} classes.")
    
    # 2. PyTorch Dataset
    class SeaDronesSeeDataset(Dataset):
        def __init__(self, annotations, images, cat_to_idx):
            self.features = []
            self.labels = []
            for ann in annotations:
                img_id = ann["image_id"]
                img = images[img_id]
                bbox = ann["bbox"] # [x, y, w, h]
                
                # Feature Engineering: aspect ratios and scale properties
                w_norm = bbox[2] / img["width"]
                h_norm = bbox[3] / img["height"]
                aspect_ratio = bbox[2] / (bbox[3] + 1e-6)
                
                # Image center coordinates
                cx_norm = (bbox[0] + bbox[2]/2.0) / img["width"]
                cy_norm = (bbox[1] + bbox[3]/2.0) / img["height"]
                center_dist = np.sqrt((cx_norm - 0.5)**2 + (cy_norm - 0.5)**2)
                
                # Crop relative area
                rel_area = (bbox[2] * bbox[3]) / (img["width"] * img["height"])
                
                feat = [w_norm, h_norm, aspect_ratio, cx_norm, cy_norm, center_dist, rel_area]
                self.features.append(feat)
                self.labels.append(cat_to_idx[ann["category_id"]])
                
            self.features = torch.tensor(self.features, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
            
    # 3. Model Definition (MLP with Softplus output for evidence)
    class EDLClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
            
        def forward(self, x):
            logits = self.net(x)
            # Guarantee evidence is positive using Softplus
            evidence = torch.functional.F.softplus(logits)
            return evidence
            
    # 4. EDL Loss (Dirichlet NLL + KL Divergence regularizer)
    def edl_loss(alpha, y_onehot, epoch_idx, num_classes, kl_annealing_epochs=10):
        S = torch.sum(alpha, dim=1, keepdim=True)
        # NLL loss under Dirichlet
        loss_nll = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        # KL regularizer to penalize incorrect class evidence
        alp_tilde = y_onehot + (1.0 - y_onehot) * alpha
        kl_alpha = torch.ones((1, num_classes), device=alpha.device)
        
        # Dirichlet KL equation
        sum_alp_tilde = torch.sum(alp_tilde, dim=1, keepdim=True)
        first_term = torch.lgamma(sum_alp_tilde) - torch.lgamma(torch.sum(kl_alpha, dim=1, keepdim=True))
        second_term = torch.sum(torch.lgamma(kl_alpha) - torch.lgamma(alp_tilde), dim=1, keepdim=True)
        third_term = torch.sum((alp_tilde - 1.0) * (torch.digamma(alp_tilde) - torch.digamma(sum_alp_tilde)), dim=1, keepdim=True)
        
        loss_kl = (first_term + second_term + third_term).squeeze()
        
        # Anneal KL coefficient
        beta = min(1.0, epoch_idx / kl_annealing_epochs)
        total_loss = torch.mean(loss_nll + beta * loss_kl)
        return total_loss

    # Split dataset into train/val
    dataset = SeaDronesSeeDataset(annotations, images, cat_to_idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    model = EDLClassifier(input_dim=7, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Beginning PyTorch training loops...")
    for epoch in range(1, 6):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            evidence = model(x)
            alpha = evidence + 1.0
            y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
            loss = edl_loss(alpha, y_onehot, epoch, num_classes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        # Validation Eval
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []
        
        with torch.no_grad():
            for x, y in val_loader:
                evidence = model(x)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                
                val_probs.append(probs.numpy())
                val_preds.append(torch.argmax(probs, dim=1).numpy())
                val_targets.append(y.numpy())
                
        val_probs = np.concatenate(val_probs, axis=0)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        acc = np.mean(val_preds == val_targets)
        ece = calculate_ece(val_probs, val_targets)
        
        print(f"Epoch {epoch}/5 | Train Loss: {train_loss/train_size:.4f} | Val Acc: {acc:.3f} | Val ECE: {ece:.4f}")
        
    # Save weights
    torch.save(model.state_dict(), "edl_weights.pth")
    print("Saved model weights to: edl_weights.pth")


def run_numpy_pipeline(json_path):
    """Fully functional NumPy implementation of the training pipeline (Fallback)."""
    print("\n" + "="*80)
    print("RUNNING PIPELINE USING NUMPY GRADIENT DESCENT (FALLBACK)")
    print("="*80)
    
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    categories = sorted(list({ann["category_id"] for ann in annotations}))
    num_classes = len(categories)
    cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(categories)}
    
    # Feature extraction
    X = []
    y = []
    for ann in annotations:
        img_id = ann["image_id"]
        img = images[img_id]
        bbox = ann["bbox"]
        
        # Features: Aspect ratio and scale
        w_norm = bbox[2] / img["width"]
        h_norm = bbox[3] / img["height"]
        aspect_ratio = bbox[2] / (bbox[3] + 1e-6)
        cx_norm = (bbox[0] + bbox[2]/2.0) / img["width"]
        cy_norm = (bbox[1] + bbox[3]/2.0) / img["height"]
        center_dist = np.sqrt((cx_norm - 0.5)**2 + (cy_norm - 0.5)**2)
        rel_area = (bbox[2] * bbox[3]) / (img["width"] * img["height"])
        
        X.append([1.0, w_norm, h_norm, aspect_ratio, cx_norm, cy_norm, center_dist, rel_area]) # include bias
        y.append(cat_to_idx[ann["category_id"]])
        
    X = np.array(X)
    y = np.array(y)
    
    # Train/Val Split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Weight matrices initialization for 5 output units (evidence logits)
    weights = np.random.normal(0, 0.1, (X.shape[1], num_classes))
    lr = 0.05
    epochs = 5
    
    print(f"NumPy training initialized on {len(X_train)} train samples, {len(X_val)} validation samples.")
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        loss_epoch = 0.0
        
        for i in range(len(X_train)):
            x_i = X_train[i]
            target_class = y_train[i]
            
            # Forward: Softplus evidence mapping
            logits = x_i @ weights
            # Softplus: ln(1 + e^x)
            evidence = np.log(1.0 + np.exp(np.clip(logits, -20, 20)))
            alpha = evidence + 1.0
            S = np.sum(alpha)
            
            # Loss calculations (NLL and KL)
            # Dirichlet NLL loss gradient w.r.t logits
            y_onehot = np.zeros(num_classes)
            y_onehot[target_class] = 1.0
            
            loss_nll = np.sum(y_onehot * (digamma(S) - digamma(alpha)))
            loss_epoch += loss_nll
            
            # Evidential backpropagation proxy: compute gradients
            d_logits = np.zeros(num_classes)
            sigmoid_logits = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20))) # derivative of softplus
            
            for k in range(num_classes):
                # Gradient of NLL w.r.t evidence_k
                d_nll_d_ev = digamma(S) - digamma(alpha[k])
                if k == target_class:
                    d_nll_d_ev -= (1.0 / alpha[k]) # digamma gradient adjustment
                d_logits[k] = d_nll_d_ev * sigmoid_logits[k]
                
            # Gradient descent step
            weights -= lr * np.outer(x_i, d_logits)
            
        # Validation step
        val_probs = []
        val_preds = []
        for i in range(len(X_val)):
            logits = X_val[i] @ weights
            evidence = np.log(1.0 + np.exp(np.clip(logits, -20, 20)))
            alpha = evidence + 1.0
            probs = alpha / np.sum(alpha)
            val_probs.append(probs)
            val_preds.append(np.argmax(probs))
            
        val_probs = np.array(val_probs)
        val_preds = np.array(val_preds)
        
        acc = np.mean(val_preds == y_val)
        ece = calculate_ece(val_probs, y_val)
        
        print(f"Epoch {epoch}/5 | Train Loss: {loss_epoch/len(X_train):.4f} | Val Acc: {acc:.3f} | Val ECE: {ece:.4f}")
        
    np.savez(MODEL_WEIGHTS_PATH, weights=weights)
    print(f"Saved NumPy training weights to: {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_JSON):
        print(f"[Info] COCO annotation file not found at {DATASET_JSON}. Running EDA first...")
        from eda_seadronessee import run_eda
        run_eda(DATASET_JSON)
        
    try:
        import torch
        train_pytorch(DATASET_JSON)
    except ImportError:
        run_numpy_pipeline(DATASET_JSON)
