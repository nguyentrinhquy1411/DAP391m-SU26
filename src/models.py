import numpy as np

# Try importing torch-related modules
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

def calculate_ece(probs, labels, num_bins=10):
    """Calculates the Expected Calibration Error (ECE) for evaluation."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
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


if TORCH_AVAILABLE:
    class EDLClassifier(nn.Module):
        """Evidential Deep Learning classifier with Softplus evidence output."""
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
            evidence = F.softplus(logits)
            return evidence

    class EvidentialCNNClassifier(nn.Module):
        """Evidential CNN Classifier using a MobileNetV3 backbone (or lightweight custom CNN fallback)."""
        def __init__(self, num_classes):
            super().__init__()
            try:
                import torchvision.models as models
                try:
                    self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                except AttributeError:
                    self.backbone = models.mobilenet_v3_small(pretrained=True)
                in_features = self.backbone.classifier[3].in_features
                self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
                self.is_fallback = False
            except Exception as e:
                print(f"[Info] CNN backbone loading failed ({e}). Falling back to custom lightweight CNN.")
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )
                self.is_fallback = True

        def forward(self, x):
            logits = self.backbone(x)
            evidence = F.softplus(logits)
            return evidence

    class SoftmaxClassifier(nn.Module):
        """Standard Softmax classifier baseline."""
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
            
        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    def edl_loss(alpha, y_onehot, epoch_idx, num_classes, kl_annealing_epochs=10):
        """
        Evidential Dirichlet NLL + annealed KL regulariser.
        Ref: Sensoy et al. NeurIPS 2018.
        """
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
        
        loss_kl = (first_term + second_term + third_term).reshape(-1)
        
        # Anneal KL coefficient
        beta = min(1.0, epoch_idx / kl_annealing_epochs)
        total_loss = torch.mean(loss_nll + beta * loss_kl)
        return total_loss
else:
    # Dummy placeholder definitions in case torch isn't available
    class EDLClassifier(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available.")
            
    class EvidentialCNNClassifier(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available.")
            
    class SoftmaxClassifier(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available.")
            
    def edl_loss(*args, **kwargs):
        raise ImportError("PyTorch is not available.")
