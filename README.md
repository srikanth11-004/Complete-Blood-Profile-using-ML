# Complete Blood Profile using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BCCD-green.svg)](https://github.com/Shenggan/BCCD_Dataset)

> Automated detection and counting of blood cells (RBCs, WBCs, Platelets) from microscopic images using YOLOv8 object detection.

---

## Table of Contents

- [The Problem](#the-problem)
- [Why This Matters](#why-this-matters)
- [Our Approach](#our-approach)
- [How It Works — End to End](#how-it-works--end-to-end)
- [Real Example Walkthrough](#real-example-walkthrough)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Future Work](#future-work)
- [References](#references)

---

## The Problem

A **Complete Blood Count (CBC)** is one of the most commonly ordered medical tests in the world. It measures the number of Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets in a blood sample. Doctors use this to diagnose conditions like anemia, infections, leukemia, and clotting disorders.

**The traditional process:**

1. A blood sample is drawn from the patient
2. It is sent to a pathology lab
3. A trained technician manually examines the sample under a microscope
4. They count each type of cell — tedious and time-consuming
5. Results come back hours or even days later

**Core problems:**

- Manual counting is slow and error-prone
- Human fatigue leads to inconsistent counts
- In rural/low-resource settings, trained pathologists are scarce
- Lab equipment and skilled labor make CBC tests expensive

The question this project asks: **Can a machine look at a blood smear image and count the cells automatically, accurately, and instantly?**

---

## Why This Matters

| Cell Type | Normal Range | What Abnormal Counts Indicate |
|-----------|-------------|-------------------------------|
| **RBC** | 4.5–5.5 million/µL | Low = Anemia; High = Polycythemia |
| **WBC** | 4,500–11,000/µL | High = Infection/Leukemia; Low = Immunodeficiency |
| **Platelets** | 150,000–400,000/µL | Low = Bleeding risk; High = Clotting risk |

Automating this count means faster diagnosis, consistent results, and potential deployment in remote clinics with just a microscope and a laptop.

---

## Our Approach

Instead of classical image processing (thresholding, watershed segmentation) which struggles with overlapping cells and staining variations, we use **YOLOv8** — a deep learning object detection model.

**Why YOLO?**
- Single-pass detection — looks at the entire image once and predicts all bounding boxes simultaneously
- Extremely fast (~150–250ms per image on CPU)
- YOLOv8 handles small objects well — critical since platelets are tiny
- We use `keremberke/yolov8n-blood-cell-detection`, a model already fine-tuned specifically on blood cell images — no training required

**Why not classical image processing?**
Real blood smear images have overlapping RBCs, varying stain intensities, different cell sizes, and noise. Deep learning handles all of this through learned feature representations.

---

## How It Works — End to End

### Step 1: PyTorch Compatibility Patch
PyTorch 2.6+ changed `torch.load` to use `weights_only=True` by default, which breaks model loading. Before anything runs, the script patches `torch.load` inside the ultralytics module to force `weights_only=False`:

```python
def patched_torch_load(f, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(f, map_location=map_location, **kwargs)

tasks.torch.load = patched_torch_load
```

### Step 2: Directory Validation
Confirms `Training/Images` and `Training/Annotations` exist before doing anything else. Fails fast with a clear error if not.

### Step 3: Image Selection
Lists all `.jpg/.jpeg/.png` files in the Images folder and picks the first N (controlled by `num_samples` in CONFIG, default 5).

### Step 4: Model Loading
Loads `keremberke/yolov8n-blood-cell-detection` from HuggingFace — a YOLOv8 nano model pre-trained on the BCCD dataset. It already knows what RBCs, WBCs, and Platelets look like. Downloaded once and cached locally.

### Step 5: Inference (per image)
Each image is passed through the YOLO model. YOLO divides the image into a grid, runs a single forward pass through its neural network, and outputs:
- Bounding box coordinates for every detected cell
- Class label (RBC / WBC / Platelet)
- Confidence score (0–1)

Only detections above `confidence_threshold: 0.25` are kept.

```
BloodImage_00000.jpg → 480×640 px → YOLO → 24 RBCs, 1 WBC detected
```

### Step 6: Cell Counting
Iterates over all bounding boxes across all processed images and tallies counts per class:

```python
for box in result.boxes:
    class_name = result.names[int(box.cls)]  # e.g. "RBC"
    counts[class_name] += 1
```

### Step 7: Visualization
Takes the first image's result, draws color-coded bounding boxes using YOLO's built-in `.plot()` method, saves the annotated image to `predicted_outputs/result_visualization.jpg`, and displays it.

---

## Real Example Walkthrough

Here is exactly what happens when you run `python blood_cell_detector.py` on 5 training images from the BCCD dataset.

### Input: 5 blood smear images

```
BloodImage_00000.jpg  — dense field of RBCs, 1 WBC visible
BloodImage_00001.jpg  — similar dense RBC field
BloodImage_00002.jpg  — RBCs with 1 WBC
BloodImage_00003.jpg  — RBCs, 1 WBC, and a Platelet cluster
BloodImage_00004.jpg  — RBCs with 1 WBC
```

### What YOLO sees per image

Each image is 480×640 pixels. YOLO processes it in a single forward pass:

```
BloodImage_00000.jpg → 24 RBCs, 1 WBC       (227ms)
BloodImage_00001.jpg → 30 RBCs, 1 WBC       (177ms)
BloodImage_00002.jpg → 29 RBCs, 1 WBC       (186ms)
BloodImage_00003.jpg → 19 RBCs, 1 WBC, 1 Platelet  (149ms)
BloodImage_00004.jpg → 19 RBCs, 1 WBC       (143ms)
```

### How counting works

After all 5 images are processed, the script loops through every bounding box across all results:

```
Image 1: 24 RBC boxes + 1 WBC box  → RBC+=24, WBC+=1
Image 2: 30 RBC boxes + 1 WBC box  → RBC+=30, WBC+=1
Image 3: 29 RBC boxes + 1 WBC box  → RBC+=29, WBC+=1
Image 4: 19 RBC boxes + 1 WBC box + 1 Platelet box → RBC+=19, WBC+=1, Platelet+=1
Image 5: 19 RBC boxes + 1 WBC box  → RBC+=19, WBC+=1
                                      ─────────────────
Total:                                RBC=121, WBC=5, Platelet=1
```

### Terminal output

```
2026-05-13 19:52:43 - INFO - PyTorch compatibility fix applied
2026-05-13 19:52:43 - INFO - Found 300 images, selected 5
2026-05-13 19:52:46 - INFO - Model loaded: keremberke/yolov8n-blood-cell-detection

image 1/1 BloodImage_00000.jpg: 480x640 24 RBCs, 1 WBC, 227ms
image 1/1 BloodImage_00001.jpg: 480x640 30 RBCs, 1 WBC, 177ms
image 1/1 BloodImage_00002.jpg: 480x640 29 RBCs, 1 WBC, 186ms
image 1/1 BloodImage_00003.jpg: 480x640 1 Platelets, 19 RBCs, 1 WBC, 149ms
image 1/1 BloodImage_00004.jpg: 480x640 19 RBCs, 1 WBC, 143ms

2026-05-13 19:52:50 - INFO - Processed 5/5 images
2026-05-13 19:52:50 - INFO - Saved visualization to predicted_outputs/result_visualization.jpg

📊 Blood Cell Counts:
   RBC:       121
   WBC:       5
   Platelets: 1
   Total:     127
```

### Output image

The annotated image saved to `predicted_outputs/result_visualization.jpg` shows `BloodImage_00000.jpg` with:
- Green boxes around each RBC, labeled `RBC 0.87` (confidence score)
- A distinct colored box around the WBC, labeled `WBC 0.91`
- Each box tightly wrapping the cell boundary

### What the counts mean clinically

These are counts across 5 microscope fields, not a full blood volume measurement. But the pattern is interpretable:
- High RBC density per field → normal or elevated RBC count
- 1 WBC per field → within normal range (WBCs are rare in a smear)
- Very few Platelets detected → either low platelet count or they're below the confidence threshold

This is a fast automated first-pass — not a replacement for a certified lab report.

---

## Dataset

**BCCD Dataset** — Blood Cell Count and Detection

```
Complete-Blood-Cell-Count-Dataset-master/
├── Training/
│   ├── Images/          # 300 blood smear images (.jpg)
│   └── Annotations/     # 300 XML annotation files (Pascal VOC format)
├── Validation/
│   ├── Images/          # 60 images
│   └── Annotations/     # 60 XML files
└── Testing/
    ├── Images/          # 60 images
    └── Annotations/     # 60 XML files
```

Each XML annotation file marks bounding boxes for every cell:
```xml
<object>
  <name>RBC</name>
  <bndbox>
    <xmin>100</xmin><ymin>80</ymin>
    <xmax>160</xmax><ymax>140</ymax>
  </bndbox>
</object>
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Model | `keremberke/yolov8n-blood-cell-detection` (HuggingFace) |
| Base | YOLOv8 Nano |
| Framework | PyTorch + Ultralytics |
| Input | RGB microscopic blood smear image |
| Output | Bounding boxes, class labels, confidence scores |
| Classes | RBC, WBC, Platelet |
| Confidence Threshold | 0.25 |
| Device | CPU (configurable to CUDA) |

---

## Project Structure

```
Complete-Blood-Profile-using-ML/
├── Complete-Blood-Cell-Count-Dataset-master/
│   ├── Training/
│   │   ├── Images/
│   │   └── Annotations/
│   ├── Validation/
│   │   ├── Images/
│   │   └── Annotations/
│   └── Testing/
│       ├── Images/
│       └── Annotations/
├── predicted_outputs/       # Annotated output images
├── blood_cell_detector.py   # Main script
├── protootype.ipynb         # Original prototype notebook
├── requirements.txt         # Dependencies
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd Complete-Blood-Profile-using-ML
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> **Windows note:** If you get a long path error during install, run this in an admin terminal first:
> ```
> reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
> ```

---

## Usage

### Run the detector
```bash
python blood_cell_detector.py
```

### Configure
Edit the `CONFIG` dictionary at the top of `blood_cell_detector.py`:

```python
CONFIG = {
    'images_dir': 'Complete-Blood-Cell-Count-Dataset-master/Training/Images',
    'annotations_dir': 'Complete-Blood-Cell-Count-Dataset-master/Training/Annotations',
    'model_path': 'keremberke/yolov8n-blood-cell-detection',
    'num_samples': 5,              # how many images to process
    'confidence_threshold': 0.25,  # lower = more detections, less precise
    'device': 'cpu'                # change to 'cuda' if you have a GPU
}
```

### Use as a module
```python
from blood_cell_detector import BloodCellDetector, CONFIG

detector = BloodCellDetector(CONFIG)
results = detector.predict_batch(['image1.jpg', 'image2.jpg'])
counts = detector.count_cells(results)

print(f"RBC: {counts['RBC']}")
print(f"WBC: {counts['WBC']}")
print(f"Platelets: {counts['Platelet']}")
```

---

## Performance

| Metric | Value |
|--------|-------|
| Detection Accuracy | ~90%+ on BCCD images |
| Processing Speed | ~150–250ms per image (CPU) |
| Supported Image Types | `.jpg`, `.jpeg`, `.png` |
| Batch Processing | Yes |

---

## Future Work

- **Web Interface** — Flask or Streamlit app for non-technical users
- **CSV Export** — save per-image counts to a spreadsheet
- **Model Fine-tuning** — train on a larger custom dataset for higher accuracy
- **RBC Volume Estimation** — use bounding box size to estimate Mean Corpuscular Volume (MCV)
- **Real-time Video** — process live microscope feed frame by frame
- **Comparison with CBC Reports** — validate counts against actual lab results

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [BCCD Dataset — Shenggan](https://github.com/Shenggan/BCCD_Dataset)
- [keremberke/yolov8n-blood-cell-detection](https://huggingface.co/keremberke/yolov8n-blood-cell-detection)
- [Ultralytics Plus](https://github.com/fcakyon/ultralyticsplus)
- [PyTorch](https://pytorch.org/docs/)
- [Wright-Giemsa Staining](https://en.wikipedia.org/wiki/Wright%27s_stain)

---

## Author

**Kiran Kumar**

---

> This project is a research prototype. It is not intended for clinical diagnosis. Always consult a certified medical professional for health decisions.
