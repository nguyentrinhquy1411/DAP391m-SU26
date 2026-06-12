import numpy as np

class EvidentialClassifierSimulator:
    """
    Simulates Evidential Deep Learning (EDL) output for posture classification.
    Postures: 0: Drowning, 1: Floating, 2: Swimming, 3: Life-Jacket-Floater
    """
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.risk_weights = np.array([1.0, 0.4, 0.2, 0.1])
        
    def estimate(self, true_class, distance, occluded=False):
        """Generates Dirichlet evidence parameters based on target distance and wave occlusion."""
        evidence = np.zeros(self.num_classes)
        
        if occluded:
            # Wave occlusion introduces high-entropy, low evidence
            evidence += np.random.uniform(0.05, 0.2, self.num_classes)
        else:
            # Evidence decays with distance (inverse relationship)
            max_ev = max(0.5, 12.0 - 0.05 * distance)
            evidence += 0.05 # small base noise
            evidence[true_class] += max_ev
            evidence += np.random.uniform(0.0, 0.1, self.num_classes)
            
        alpha = evidence + 1.0
        S = np.sum(alpha)
        
        # Subjective Logic formalism
        beliefs = evidence / S
        u = self.num_classes / S
        probs = alpha / S
        
        return beliefs, u, probs

# Alias for backwards compatibility
EvidentialClassifier = EvidentialClassifierSimulator
