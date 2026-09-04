# TROMCAP - Traffic Object Recognition and Motion Capture

A computer vision project for object detection, action recognition, and tracking in traffic video analysis using deep learning models.

## Project Overview

TROMCAP is designed to process video data for detecting vehicles/objects, recognizing actions, and tracking movements across frames. The project integrates YOLO-based detection with action recognition and tracking capabilities.

---

## 📁 Project Directory Structure

```
tromcap/
├── src/                          # Main source code directory
│   ├── api/                      # API endpoints and HTTP handlers
│   ├── core/                     # Core business logic and algorithms
│   ├── database/                 # Database models and operations
│   ├── pipeline/                 # Data processing pipelines
│   ├── services/                 # Service modules
│   │   └── object_detect.py      # Object detection service
│   ├── utils/                    # Utility functions and helpers
│   └── instruction.md            # Source code documentation
│
├── models/                       # Pre-trained and custom ML models
│   ├── action_recognition/       # Action recognition model files
│   ├── detection/                # Object detection model files
│   └── tracking/                 # Tracking model files
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed/cleaned data
│   ├── images/                   # Image dataset
│   ├── videos/                   # Video files for processing
│   │   ├── video_17.mp4
│   │   ├── video_18.mp4
│   │   └── ... (video_19-31.mp4)
│   ├── annotations/              # Annotation files
│   │   ├── action/               # Action annotations
│   │   └── detection/            # Detection annotations
│   ├── labels/                   # Object labels and metadata
│   ├── samples/                  # Sample data for testing
│   └── instruction.md            # Data documentation
│
├── configs/                      # Configuration files
│                                 # (Model configs, pipeline configs, etc.)
│
├── scripts/                      # Utility and automation scripts
│                                 # (Training, preprocessing, deployment scripts)
│
├── tests/                        # Unit and integration tests
│                                 # (Test suites for all modules)
│
├── docs/                         # Documentation files
│                                 # (API docs, architecture, usage guides)
│
├── deployment/                   # Deployment configurations
│   ├── docker/                   # Docker containerization
│   ├── cloud_config/             # Cloud deployment configs
│   └── edge_config/              # Edge device deployment configs
│
├── main.py                       # Main entry point of the application
├── test.py                       # Test runner script
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependencies (uv package manager)
├── .python-version              # Python version specification
├── .gitignore                   # Git ignore rules
└── README.md                    # This file

```

---

## 🛠️ Technology Stack

### Core Dependencies
- **Python**: >= 3.12.10
- **Ultralytics**: >= 8.4.110 (YOLO object detection)
- **Ollama**: >= 0.6.2 (LLM integration)

### Machine Learning Components
- Object Detection (YOLO-based models)
- Action Recognition
- Object Tracking

### Development Tools
- **Package Manager**: uv (modern Python package manager)
- **Testing**: Custom test.py runner

---

## 📋 Directory Details

### `src/` - Source Code
Contains the main application logic organized by functionality:
- **api/**: REST API or interface endpoints
- **core/**: Core algorithms and business logic
- **database/**: Data storage and retrieval operations
- **pipeline/**: Data processing and workflow pipelines
- **services/**: Service modules for specific tasks (e.g., object_detect.py)
- **utils/**: Helper functions and utilities

### `models/` - ML Models
Houses pre-trained and trained model files:
- **action_recognition/**: Models for action/behavior classification
- **detection/**: Object detection models (YOLO weights, etc.)
- **tracking/**: Multi-object tracking models

### `data/` - Dataset Directory
Contains all input, output, and annotation data:
- **raw/**: Unprocessed source data
- **processed/**: Cleaned and processed data
- **images/**: Image dataset
- **videos/**: Video files for analysis
- **annotations/**: Ground truth labels and annotations
- **labels/**: Object class definitions and metadata
- **samples/**: Sample data for quick testing

### `configs/` - Configuration Files
Stores configuration for models, pipelines, and parameters.

### `scripts/` - Automation Scripts
Utility scripts for:
- Data preprocessing
- Model training
- Evaluation
- Deployment utilities

### `tests/` - Test Suite
Unit tests and integration tests for all modules.

### `docs/` - Documentation
API documentation, architecture diagrams, and usage guides.

### `deployment/` - Deployment Configurations
- **docker/**: Containerized deployment
- **cloud_config/**: Cloud platform deployment (AWS, GCP, Azure, etc.)
- **edge_config/**: Edge device deployment (Jetson, Raspberry Pi, etc.)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12.10 or higher
- Virtual environment (`.venv/`)

### Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .
# or with uv:
uv sync
```

### Running the Project
```bash
# ByteTrack tuned for short missed detections (default)
python main.py data/videos/video_11.mp4

# Appearance/ReID profile for deformation and occlusion (slower)
python main.py data/videos/video_11.mp4 --tracker configs/custom_botsort.yaml

# Run tests
python test.py
```

### Tracker Benchmark

Benchmark ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTrack và TrackTrack trên toàn bộ video:

```powershell
.\.venv\Scripts\python.exe -m experiments.tracker_benchmark.run --videos data/videos
```

Xem hướng dẫn và ý nghĩa metric tại `experiments/tracker_benchmark/README.md`.

### Person-bag rule engine

Mặc định pipeline video chạy thêm rule engine có trạng thái để:

- nối `raw_track_id` của tracker vào `entity_id` ổn định;
- duy trì quan hệ holder của từng bag qua các đoạn mất detection ngắn;
- tính chuyển động đã làm mượt theo thời gian thực;
- phát hiện chuỗi `approach → contact → holder transfer → escape`;
- đánh dấu theo sự kiện: `normal`, `possible_victim`, `possible_suspect`,
  `victim`, `suspect`.

Chạy với BoT-SORT/ReID và cấu hình rule mặc định:

```powershell
.\.venv\Scripts\python.exe main.py data/videos/video_27.mp4 `
  --tracker configs/custom_botsort.yaml `
  --rules-config configs/snatch_rules.yaml
```

Chỉ chạy detector/tracker, không chạy analytics:

```powershell
.\.venv\Scripts\python.exe main.py data/videos/video_27.mp4 --disable-analytics
```

Kết quả analytics được lưu trong các bảng SQLite:

- `entity_observations`: canonical ID, raw ID và cờ observed/reliable;
- `bag_person_relations`: holder, baseline holder, candidate và evidence;
- `snatch_events`: trạng thái, score, victim, suspect và evidence;
- `person_roles`: role của person theo từng frame.

Các ngưỡng đều dùng giây hoặc tọa độ chuẩn hóa thay vì số frame/pixel cố định.
Chi tiết module, state machine và hướng dẫn tuning nằm trong
[`script_rule_engine.md`](script_rule_engine.md).

---

## 📝 Configuration

- Python version: Defined in `.python-version`
- Dependencies: Listed in `pyproject.toml`
- Locked versions: Specified in `uv.lock`

---

## 🔍 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `test.py` | Test runner |
| `configs/snatch_rules.yaml` | Rule thresholds and evidence weights |
| `src/analytics/` | Canonical identity, relation, motion and event engine |
| `pyproject.toml` | Project metadata and dependencies |
| `.gitignore` | Git ignore rules (excludes data/) |
| `src/instruction.md` | Source code documentation |
| `data/instruction.md` | Data documentation |

---

## 📚 Documentation

- **Source Code Docs**: See `src/instruction.md`
- **Data Docs**: See `data/instruction.md`
- **Full Documentation**: See `docs/` directory

---

## 🔧 Development

The project uses a modular architecture enabling:
- Independent development of detection, tracking, and action recognition
- Flexible pipeline configuration
- Easy deployment to multiple environments (cloud, edge)

---

## 📦 Project Structure Philosophy

```
Configuration → Pipeline → Models → Output
                    ↓
                Services (Detection, Tracking, Recognition)
```

---

*Last Updated: 2026-08-26*
