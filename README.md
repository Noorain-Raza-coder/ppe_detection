# Person & PPE Detection Pipeline

A two-stage Computer Vision pipeline for detecting **persons and Personal Protective Equipment (PPE)** from images.

The pipeline first detects people in the complete image and then performs PPE detection separately on each detected person crop. The PPE predictions are subsequently mapped back to the original image coordinates and visualized using OpenCV.

---

## 1. Problem Statement

The objective of this project is to build an object detection pipeline capable of detecting:

* Persons
* Boots
* Gloves
* Hard-hat
* Mask
* PPE-suit
* Vest

The solution uses two separate YOLO object detection models:

1. A **Person Detection Model** trained on complete images.
2. A **PPE Detection Model** trained on cropped person images.

During inference, the two models are combined into a single pipeline.

```text id="9n0tps"
Input Image
     │
     ▼
Person Detection Model
     │
     ▼
Person Bounding Boxes
     │
     ▼
Person Crops
     │
     ▼
PPE Detection Model
     │
     ▼
PPE Predictions
     │
     ▼
Map Predictions Back
to Original Image
     │
     ▼
Annotated Output Image
```

This approach allows the PPE detector to focus on individual people rather than detecting PPE directly from the complete scene.

---

# 2. Project Structure

```text id="0j4xak"
project/
│
├── datasets/
│   │
│   ├── pascalVOC_dataset/
│   │   └── Dataset provided with the assignment
│   │
│   ├── voc_to_yolo_dataset/
│   │   └── Dataset converted from Pascal VOC to YOLO format
│   │
│   ├── yolo_augmented_dataset/
│   │   └── Augmented and split dataset downloaded from Roboflow
│   │
│   ├── yolo_person_only_dataset/
│   │   └── Dataset prepared for person detection
│   │
│   ├── yolo_ppe_dataset/
│   │   └── Dataset prepared for PPE detection
│   │
│   └── test_dataset/
│       └── Images collected for testing the complete pipeline
│
├── src/
│   │
│   ├── inference.py
│   │   └── Runs the complete person + PPE inference pipeline
│   │
│   ├── pascalVOC_to_yolo.py
│   │   └── Converts Pascal VOC annotations to YOLO format
│   │
│   ├── prepare_person_dataset.py
│   │   └── Prepares the person detection dataset
│   │
│   ├── prepare_ppe_dataset.py
│   │   └── Prepares the PPE detection dataset
│   │
│   ├── train_person.py
│   │   └── Trains the person detection model
│   │
│   └── train_ppe.py
│       └── Trains the PPE detection model
│
├── weights/
│   ├── person_best.pt
│   │   └── Trained person detection model
│   │
│   └── ppe_best.pt
│       └── Trained PPE detection model
│
├── config.py
│   └── Centralized application configuration
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

# 3. Environment

The project was developed and tested using a **Conda environment**.

### Python

```text
Python 3.10
```

### Main dependencies

| Package          | Version      |
| ---------------- | ------------ |
| Python           | 3.10         |
| PyTorch          | 2.12.1+cu130 |
| Ultralytics YOLO | 8.4.75       |
| OpenCV           | 4.13.0       |
| NumPy            | 2.2.6        |

The complete dependency specification is available in:

```text
requirements.txt
```

---

# 4. Installation

## Step 1 — Create the Conda environment

```bash
conda create -n ppe-detection python=3.10
```

Activate the environment:

```bash
conda activate ppe-detection
```

## Step 2 — Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Verify the environment after installation:

```bash
python --version
```

The expected Python version is:

```text
Python 3.10.x
```

---

# 5. Model Weights

The trained model weights are included in the `weights/` directory:

```text
weights/
├── person_best.pt
└── ppe_best.pt
```

### `person_best.pt`

Used to detect persons in the complete input image.

### `ppe_best.pt`

Used to detect PPE items in the cropped person images generated during inference.

---

# 6. Configuration

Application-level configuration is maintained in:

```text
config.py
```

The configuration includes values such as:

* Person detection confidence threshold
* PPE detection confidence threshold
* PPE class mapping
* Class-specific visualization colors
* Supported image extensions
* OpenCV drawing configuration

This keeps configuration separate from the main inference implementation.

---

# 7. Running the Inference Pipeline

The complete inference pipeline is implemented in:

```text
src/inference.py
```

The script accepts four command-line arguments:

```text
--input_dir
--output_dir
--person_det_model
--ppe_detection_model
```

## Basic command

From the project root, first try:

```bash
python3 src/inference.py \
    --input_dir datasets/test_dataset \
    --output_dir inference_results \
    --person_det_model weights/person_best.pt \
    --ppe_detection_model weights/ppe_best.pt
```

If the above command results in a module/import-related error, run the script as a Python module instead:

```bash
python3 -m src.inference \
    --input_dir datasets/test_dataset \
    --output_dir inference_results \
    --person_det_model weights/person_best.pt \
    --ppe_detection_model weights/ppe_best.pt
```

### Command-line arguments

| Argument                | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `--input_dir`           | Directory containing input images              |
| `--output_dir`          | Directory where annotated images will be saved |
| `--person_det_model`    | Path to the person detection model weights     |
| `--ppe_detection_model` | Path to the PPE detection model weights        |

---

# 8. Output

After successful execution, the annotated images are saved to the specified output directory.

For example:

```text
inference_results/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

The output images contain the detected objects along with their confidence scores.

---

# 9. Documentation

For a detailed understanding of **every step of the project**, including dataset preparation, annotation conversion, model training, PPE crop generation, coordinate transformation, training decisions, evaluation metrics, observations, and other technical details, please refer to the **attached project report**.

The README is intentionally limited to project overview, structure, environment setup, configuration, and execution instructions.
