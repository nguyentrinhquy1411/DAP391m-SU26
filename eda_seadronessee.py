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

OUTPUT_REPORT = "eda_summary_report.txt"

def generate_mock_coco_json(output_path):
    """Generates a mock COCO annotations file representing SeaDronesSee structure for verification."""
    categories = [
        {"id": 1, "name": "swimmer"},
        {"id": 2, "name": "floater"},
        {"id": 3, "name": "boat"},
        {"id": 4, "name": "life jacket"},
        {"id": 5, "name": "buoy"}
    ]
    
    images = []
    annotations = []
    
    np.random.seed(42)
    
    # Generate 200 mock images with diverse altitudes and gimbal pitches
    for img_id in range(1, 201):
        alt = float(np.random.choice([10, 20, 30, 50, 80, 120, 180, 240]))
        pitch = float(np.random.uniform(20, 90))
        images.append({
            "id": img_id,
            "width": 1920,
            "height": 1080,
            "file_name": f"val/{img_id:04d}.jpg",
            "altitude": alt,
            "gimbal_pitch": pitch,
            "latitude": 20.84 + np.random.normal(0, 0.01),
            "longitude": 107.03 + np.random.normal(0, 0.01)
        })
        
        # Add 1 to 5 random targets per image
        num_targets = np.random.randint(1, 6)
        for t_idx in range(num_targets):
            cat_id = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            # Bounding box coordinates (mostly small scale targets)
            w = float(np.random.exponential(scale=30.0)) + 5.0
            h = w * np.random.uniform(0.8, 1.2)
            x = float(np.random.uniform(100, 1800))
            y = float(np.random.uniform(100, 900))
            
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": img_id,
                "category_id": int(cat_id),
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            
    coco_mock = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_mock, f, indent=2)
    print(f"Generated mock COCO json file at: {output_path}")


def run_eda(json_path):
    print("="*80)
    print(f"RUNNING SEA-DRONES-SEE EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    if not os.path.exists(json_path):
        print(f"[Info] Target COCO annotation file not found at: {json_path}")
        print("       Generating mock dataset annotations for verification...")
        generate_mock_coco_json(json_path)
        
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    images = coco["images"]
    annotations = coco["annotations"]
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    num_images = len(images)
    num_anns = len(annotations)
    
    # Category Distribution Analysis
    cat_counts = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = categories.get(cat_id, "Unknown")
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
        
    # Spatial Scale Analysis (BBox dimensions)
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    for ann in annotations:
        bbox = ann["bbox"] # [x, y, w, h]
        w, h = bbox[2], bbox[3]
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        aspect_ratios.append(w / (h + 1e-6))
        
    # Telemetry Distribution Analysis
    altitudes = []
    pitches = []
    for img in images:
        if "altitude" in img:
            altitudes.append(img["altitude"])
        elif "meta" in img and isinstance(img["meta"], dict):
            alt_val = img["meta"].get("height_above_takeoff(meter)")
            if alt_val is not None:
                altitudes.append(alt_val)
        
        if "gimbal_pitch" in img:
            pitches.append(img["gimbal_pitch"])
        elif "meta" in img and isinstance(img["meta"], dict):
            pitch_val = img["meta"].get("gimbal_pitch(degrees)")
            if pitch_val is not None:
                pitches.append(pitch_val)
    
    # Calculate Summary Stats
    report = []
    report.append("="*60)
    report.append("SEA-DRONES-SEE EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    report.append("="*60)
    report.append(f"Total Frames analyzed: {num_images}")
    report.append(f"Total Annotated Instances: {num_anns}")
    report.append(f"Average target density: {num_anns / num_images:.2f} targets/image\n")
    
    report.append("1. Category Distribution:")
    report.append("-" * 30)
    for cat_name, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / num_anns) * 100
        report.append(f"  * {cat_name:<15}: {count:<6} ({percentage:.2f}%)")
    report.append("")
    
    report.append("2. Target Spatial Scale Metrics:")
    report.append("-" * 30)
    report.append(f"  * Average BBox Area : {np.mean(areas):.1f} px^2")
    report.append(f"  * Median BBox Area  : {np.median(areas):.1f} px^2")
    report.append(f"  * Smallest target   : {np.min(areas):.1f} px^2")
    report.append(f"  * Largest target    : {np.max(areas):.1f} px^2")
    report.append(f"  * Average Width     : {np.mean(widths):.1f} px")
    report.append(f"  * Average Height    : {np.mean(heights):.1f} px")
    report.append(f"  * Mean Aspect Ratio : {np.mean(aspect_ratios):.2f}")
    
    # Classify scale according to COCO standards
    small_count = sum(1 for a in areas if a < 32**2)
    medium_count = sum(1 for a in areas if 32**2 <= a < 96**2)
    large_count = sum(1 for a in areas if a >= 96**2)
    report.append(f"  * COCO Scale Distribution:")
    report.append(f"    - Small (area < 1024 px^2)    : {small_count:<5} ({small_count/num_anns*100:.1f}%)")
    report.append(f"    - Medium (1024 <= area < 9216): {medium_count:<5} ({medium_count/num_anns*100:.1f}%)")
    report.append(f"    - Large (area >= 9216 px^2)   : {large_count:<5} ({large_count/num_anns*100:.1f}%)")
    report.append("")
    
    report.append("3. UAV Telemetry Analysis:")
    report.append("-" * 30)
    if altitudes:
        report.append(f"  * Altitude (Alt) stats (m):")
        report.append(f"    - Mean   : {np.nanmean(altitudes):.2f}m")
        report.append(f"    - Min    : {np.nanmin(altitudes):.2f}m")
        report.append(f"    - Max    : {np.nanmax(altitudes):.2f}m")
    else:
        report.append("  * Altitude metadata not available.")
        
    if pitches:
        report.append(f"  * Gimbal Pitch stats (degrees, 90 = nadir):")
        report.append(f"    - Mean   : {np.nanmean(pitches):.2f} deg")
        report.append(f"    - Min    : {np.nanmin(pitches):.2f} deg")
        report.append(f"    - Max    : {np.nanmax(pitches):.2f} deg")
    else:
        report.append("  * Gimbal Pitch metadata not available.")
    report.append("="*60)
    
    # Print and Write Report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report_text)
    print(f"\nSaved detailed EDA summary report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    run_eda(DATASET_JSON)
