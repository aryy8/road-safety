# Road Safety Automated Audit System

An AI-powered computer vision system for automated road infrastructure monitoring, safety assessment, and maintenance prioritization. This project leverages a **trained YOLO model deployed via Roboflow's serverless API** to detect **17 road safety classes**, utilizing the **"detect-count-and-visualize" workflow** for real-time processing and automated report generation.
![WhatsApp Image 2025-11-15 at 13 37 09 (1)](https://github.com/user-attachments/assets/27378382-ecf4-45d7-a991-7b7408772b47)
![WhatsApp Image 2025-11-15 at 13 36 18](https://github.com/user-attachments/assets/3f29ba67-03c0-4a57-8a8a-f7f0e398b975)
<br>
![WhatsApp Image 2025-11-15 at 13 37 09](https://github.com/user-attachments/assets/5cf3c14f-f87c-464a-9421-8d237c79d1c0)
![WhatsApp Image 2025-11-15 at 13 36 18 (2)](https://github.com/user-attachments/assets/d7ea5632-ebcc-4e7f-8631-30a2fc3eb68a)





## Key Features

- **Serverless Inference**: Cloud-based YOLO model processing via Roboflow API
- **Real-time Visualization**: Live detection counting and visualization
- **17-Class Detection**: Comprehensive road infrastructure coverage
- **Automated Reporting**: PDF reports with risk scoring and trend analysis
- **CSV Logging**: Structured detection data with unique identifiers

## Tech Stack

### Computer Vision & ML
- **YOLOv8** (via `ultralytics`) - Primary object detection framework
- **YOLOv5** (via `torch.hub`) - Vulnerable road user detection
- **PyTorch** - Deep learning backend
- **OpenCV** (`cv2`) - Video processing and visualization

### Data Processing
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **supervision** - Computer vision utilities

### Visualization & Reporting
- **Matplotlib** - Charts and statistical plots
- **ReportLab** - Structured PDF reports
- **FPDF** - Advanced PDF generation with color coding

### Dataset Management
- **Roboflow** - Dataset hosting, annotation management, and serverless API
- **YAML** - Configuration files

### Serverless API & Workflow
- **Roboflow Inference API** - Cloud-based model inference
- **Workflow Engine** - "detect-count-and-visualize" for automated processing
- **Real-time Processing** - Video analysis with live visualization
- **CSV Export** - Structured detection logging with unique IDs

##  Project Structure

```
road_safety/
├── roadeye-nf4ie/1          # Self annotated dataset
│   ├── data.yaml               # Dataset configuration
│   ├── train/                  # Training images and labels
│   ├── valid/                  # Validation images and labels 
│   └── test/                   # Test images and labels
├── runs/                       # YOLO training/inference outputs
│   └── detect/                 # Detection results
├── report_outputs/             # Basic analysis reports
├── audit_report_outputs/       # Advanced audit reports
├── risk_report_outputs/        # Risk assessment reports
├── *.pt                        # Trained YOLO model weights
├── *.py                        # Python scripts
├── *.csv                       # Detection data and analysis results
├── *.png                       # Visualization outputs
└── *.MP4                       # Video files for analysis
```




## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Setup
1. **Clone and navigate to the repository:**
   ```bash
   cd /Users/aryan/road_safety
   ```

2. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python pandas numpy matplotlib reportlab fpdf supervision inference
   ```

3. **Install PyTorch (with CUDA if available):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Core Components

### 1. Roboflow Workflow Inference (`final_eval_logs.py`)
**Primary inference method using trained YOLO model via serverless API**

**Features:**
- Real-time video processing via Roboflow workflow
- Automated detection counting and visualization
- CSV logging of all detections with unique IDs
- Live preview during processing

**Usage:**
```bash
python final_eval_logs.py
```

**Workflow:** "detect-count-and-visualize"
- Processes video through trained YOLO model
- Provides real-time counting and visualization
- Saves structured detection data to CSV

![WhatsApp Image 2025-11-15 at 12 17 45](https://github.com/user-attachments/assets/4da208ca-7ede-4003-9e12-bece3865a95e)


### 2. Multi-Model Video Inference (`multi_model_video_inference.py`)
**Alternative local inference using multiple specialized YOLO models**

**Models Used:**
- `best_markings.pt` - Road markings and lane lines
- `best_pothole.pt` - Pothole detection
- `best_sign.pt` - Traffic signs
- `vulnerable_best.pt` - Vulnerable road users (YOLOv5)

**Usage:**
```bash
python multi_model_video_inference.py --input Trial_1_Video.MP4 --output trial1_detections.mp4 --conf 0.5 --show
```

**Parameters:**
- `--input`: Input video path
- `--output`: Output video path
- `--markings`, `--pothole`, `--sign`, `--vulnerable`: Model weight paths
- `--conf`: Confidence threshold
- `--show`: Display live processing window
- `--device`: CPU/CUDA selection

### 3. Detection Analysis (`analyze_detections.py`)
Analyzes detection results to identify road infrastructure deterioration.

**Features:**
- Class distribution analysis
- Temporal trend analysis
- Area variation tracking
- Missing element detection
- Deterioration summary statistics

**Input:** `detections.csv`
**Outputs:** Statistical summaries, trend plots, and audit reports

### 4. Full Report Generation (`generate_full_report.py`)
**Enhanced PDF report generation with comprehensive analysis**

**Features:**
- Advanced statistical analysis
- Base vs present comparison (if available)
- Detailed class distribution plots
- Trend analysis across frames
- Area variation tracking for all classes

**Usage:**
```bash
python generate_full_report.py
```

### 5. Basic Report Generation (`generate_report.py`)
Creates structured PDF reports with visualizations.

**Features:**
- Class distribution charts
- Frame-by-frame trend analysis
- Area variation plots for critical elements
- Statistical summaries in tabular format

**Output:** `report_outputs/road_safety_report.pdf`

### 6. Risk Assessment (`audit_report_outputs/generate_risk_report.py`)
Advanced risk scoring system for maintenance prioritization.

**Risk Factors Analyzed:**
- Missing ratio (40% weight)
- Fading ratio (30% weight)
- Detection instability (20% weight)
- Confidence drop (10% weight)

**Risk Score Scale:** 0-100 (Low: <33, Moderate: 33-66, High: >66)

**Outputs:**
- Color-coded risk table PDF
- Per-class detail pages
- Trend analysis plots
- CSV risk scores
![WhatsApp Image 2025-11-15 at 12 38 56](https://github.com/user-attachments/assets/66cbbed6-5ad2-4c86-a095-85fe5a77845a)


## Data Formats

### Detection Results (`detections.csv`)
```csv
frame,class,confidence,x1,y1,x2,y2,width,height,bbox_area,uuid
0,vehicle,0.91357421875,2861,968,3840,1842,979,874,855646,12d27e3b-...
0,edgeline,0.732421875,1732,1339,2076,1861,344,522,179568,eade74b2-...
```

**Columns:**
- `frame`: Frame number in video
- `class`: Detected object class
- `confidence`: Detection confidence (0-1)
- `x1,y1,x2,y2`: Bounding box coordinates
- `width,height`: Bounding box dimensions
- `bbox_area`: Bounding box area (pixels²)
- `uuid`: Unique detection identifier

### Dataset Classes (17 classes)
**Trained YOLO Model Classes:**
- barrier, divider, edgeline, foothpath, grassstrip
- lane, laneline, offroad, pole, road marking
- sign board, streetlight, traffic light, vehicle
- yellow markings, zebra crossing
![WhatsApp Image 2025-11-15 at 12 41 37](https://github.com/user-attachments/assets/e180a390-4ad5-4e33-b377-06da6e8d6797)

## Usage Workflows


### 1. Primary Workflow: Roboflow Serverless API
```bash
# Step 1: Process video with trained YOLO model via Roboflow workflow
python final_eval_logs.py

# Step 2: Generate comprehensive audit report
python generate_full_report.py

# Step 3: Generate risk assessment (optional advanced analysis)
cd audit_report_outputs && python generate_risk_report.py
```

### 2. Alternative: Local Multi-Model Inference
```bash
# Step 1: Run multi-model inference with local weights
python multi_model_video_inference.py --input Trial_1_Video.MP4 --output processed_video.mp4

# Step 2: Analyze detections (ensure detections.csv exists)
python analyze_detections.py

# Step 3: Generate basic report
python generate_report.py
```

### 3. Single Image Testing
```python
from ultralytics import YOLO

# Load model
model = YOLO("best_sign.pt")

# Run inference
results = model.predict(source="speedsign.png", conf=0.5, save=True)
```

### 4. Direct Roboflow API Integration
```python
from inference import InferencePipeline

# Initialize pipeline with custom workflow
pipeline = InferencePipeline.init_with_workflow(
    api_key="Bj727w9uyjQFBN2OgnjK",
    workspace_name="aryy8",
    workflow_id="detect-count-and-visualize",
    video_reference="Trial_1_Video.MP4",
    max_fps=30
)

# Start processing
pipeline.start()
pipeline.join()
```

##  Model Training

### Using Roboflow
1. Upload dataset to Roboflow Universe
2. Train YOLOv8 models via web interface
3. Download trained weights (`.pt` files)

### Local Training
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(
    data='My-First-Project-2/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Output Analysis

### Risk Score Interpretation
- **0-33 (Green)**: Low risk - Periodic monitoring
- **33-66 (Yellow)**: Moderate risk - Schedule inspection
- **66-100 (Red)**: High risk - Immediate action required

### Key Metrics Tracked
- **Detection Count Trends**: Monitor for sudden drops
- **Area Variations**: Identify fading/marking deterioration
- **Missing Elements**: Critical safety components not detected
- **Confidence Degradation**: Model performance over time

##  Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing Models**: Ensure `.pt` files are in working directory
3. **Video Processing Slow**: Lower `--max_fps` or use CPU mode
4. **Empty Reports**: Check if `detections.csv` exists and has data

### Performance Optimization
- Use GPU for faster inference
- Adjust confidence thresholds based on use case
- Process videos in segments for large files
- Use `--device cpu` for systems without CUDA


### Development Guidelines
- Follow existing code style and structure
- Add docstrings to new functions
- Test with multiple video formats
- Update README for new features

##  License

This project is licensed under CC BY 4.0 (as per Roboflow dataset license).

## Acknowledgments

- **Roboflow** for dataset hosting and model training
- **Ultralytics** for YOLOv8 implementation
- **PyTorch** for deep learning framework
- **OpenCV** for computer vision utilities

---

**Note:** This system is designed for automated road safety auditing and should be used as a supplement to, not replacement for, human inspection and professional engineering assessment.
