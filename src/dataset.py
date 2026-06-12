import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    class TorchDataset:
        pass

class SeaDronesSeeDataset(TorchDataset):
    """
    Wraps SeaDronesSee COCO annotations into a Dataset.
    Supports both PyTorch Dataset API and standard NumPy fallback indexing.
    """
    def __init__(self, annotations, images, cat_to_idx, include_center_dist=True):
        self.features = []
        self.labels = []
        for ann in annotations:
            img_id = ann["image_id"]
            img = images[img_id]
            bbox = ann["bbox"] # [x, y, w, h]
            
            w_norm = bbox[2] / img["width"]
            h_norm = bbox[3] / img["height"]
            aspect = bbox[2] / (bbox[3] + 1e-6)
            cx_norm = (bbox[0] + bbox[2]/2.0) / img["width"]
            cy_norm = (bbox[1] + bbox[3]/2.0) / img["height"]
            rel_area = (bbox[2] * bbox[3]) / (img["width"] * img["height"])
            
            if include_center_dist:
                center_dist = np.sqrt((cx_norm - 0.5)**2 + (cy_norm - 0.5)**2)
                feat = [w_norm, h_norm, aspect, cx_norm, cy_norm, center_dist, rel_area]
            else:
                feat = [w_norm, h_norm, aspect, cx_norm, cy_norm, rel_area]
                
            self.features.append(feat)
            self.labels.append(cat_to_idx[ann["category_id"]])
            
        try:
            import torch
            self.features = torch.tensor(self.features, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        except ImportError:
            self.features = np.array(self.features, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
