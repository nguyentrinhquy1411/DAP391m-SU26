import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    class TorchDataset:
        pass
    TORCH_AVAILABLE = False


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


if TORCH_AVAILABLE:
    class SeaDronesSeeImageDataset(TorchDataset):
        """
        Loads raw images, crops the annotated bounding boxes on-the-fly,
        resizes to a fixed size (e.g. 64x64), and yields image tensors + labels.
        """
        def __init__(self, annotations, images, cat_to_idx, base_dir="data/images", target_size=(64, 64), transform=None):
            self.annotations = annotations
            self.images = images
            self.cat_to_idx = cat_to_idx
            self.base_dir = base_dir
            self.target_size = target_size
            self.transform = transform
            
            # Resolve image paths recursively to avoid walking during __getitem__
            self.resolved_paths = {}
            for img_id, img_info in images.items():
                fname = img_info["file_name"]
                found = False
                for folder in ["val", "train", "test", ""]:
                    path = os.path.join(base_dir, folder, fname) if folder else os.path.join(base_dir, fname)
                    if os.path.exists(path):
                        self.resolved_paths[img_id] = os.path.abspath(path)
                        found = True
                        break
                if not found:
                    # Fallback to recursively looking for the file under base_dir
                    for root, dirs, files in os.walk(base_dir):
                        if fname in files:
                            self.resolved_paths[img_id] = os.path.abspath(os.path.join(root, fname))
                            break
            
            # Keep only annotations for which we successfully resolved an image path
            self.valid_anns = []
            for ann in annotations:
                img_id = ann["image_id"]
                if img_id in self.resolved_paths:
                    self.valid_anns.append(ann)
                    
            if len(self.valid_anns) == 0:
                print(f"[Warning] No images resolved in '{base_dir}'. Dataset is empty!")
                
        def __len__(self):
            return len(self.valid_anns)
            
        def __getitem__(self, idx):
            ann = self.valid_anns[idx]
            img_id = ann["image_id"]
            img_path = self.resolved_paths[img_id]
            
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    bbox = ann["bbox"] # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    x1 = max(0, int(round(x)))
                    y1 = max(0, int(round(y)))
                    x2 = min(img_w, int(round(x + w)))
                    y2 = min(img_h, int(round(y + h)))
                    
                    if x2 <= x1 or y2 <= y1:
                        crop = Image.new("RGB", self.target_size, (128, 128, 128))
                    else:
                        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            except Exception:
                crop = Image.new("RGB", self.target_size, (128, 128, 128))
                
            crop = crop.resize(self.target_size, Image.Resampling.BILINEAR)
            
            if self.transform:
                crop_tensor = self.transform(crop)
            else:
                t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                crop_tensor = t(crop)
                
            label = self.cat_to_idx[ann["category_id"]]
            return crop_tensor, label
else:
    class SeaDronesSeeImageDataset(TorchDataset):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch/torchvision are not available.")
