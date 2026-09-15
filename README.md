# TROMCAP — Snatching Detection

Hệ thống phân tích video để phát hiện hành vi cướp giật dựa trên YOLO, multi-object tracking và rule engine theo thời gian cho quan hệ người–túi.

## Cấu trúc dự án

```text
.
├── configs/                         # Cấu hình detector, tracker và rule engine
├── data/                            # Video đầu vào, SQLite và kết quả sinh ra
├── models/
│   └── detection/                   # YOLO weights
├── src/                             # Main source code directory
│   ├── api/                         # API endpoints and HTTP handlers
│   ├── core/                        # Core business logic and algorithms
│   │   ├── types.py                 # Detection, TrackedObject, BBox dùng chung
│   │   ├── vision/                  # YOLO và tracker adapters
│   │   └── analytics/               # Rule engine người–túi và snatch event
│   ├── database/                    # Database models and operations (SQLite connection, schema, repositories)
│   ├── pipeline/                    # Pipeline video end-to-end
│   ├── services/                    # Public application services
│   ├── utils/                       # Visualizer và latency measurement
│   └── instruction.md               # Source code documentation
├── scripts/                         # Utility and automation scripts
├── tests/                           # Unit và integration tests
├── docs/                            # Documentation files
│                                    # (API docs, architecture, usage guides)
│
├── deployment/                      # Deployment configurations
│   ├── docker/                      # Docker containerization
│   ├── cloud_config/                # Cloud deployment configs
│   └── edge_config/                 # Edge device deployment configs
│
├── main.py                          # Main entry point of the application
├── test.py                          # Test runner script
├── pyproject.toml                   # Project metadata and dependencies
├── uv.lock                          # Locked dependencies (uv package manager)
├── .python-version                  # Python version specification
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

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


## Luồng xử lý

```text
main.py / api → pipeline → services → core
                         ├→ database
                         └→ utils
```

- `core/analytics` không phụ thuộc OpenCV, Ultralytics hoặc SQLite.
- `core/vision` chỉ bọc các thư viện detection/tracking, không chứa snatching rules.
- `pipeline` điều phối frame; `services` là ranh giới sử dụng ổn định cho caller.
- `DetectionDB` giữ API cũ và chuyển tiếp thao tác tới các repository SQLite theo trách nhiệm.

## Cài đặt

```powershell
uv sync
```

## Chạy ứng dụng

```powershell
uv run main.py data/videos/video_11.mp4
```

Chạy với BoT-SORT và rule engine:

```powershell
uv run main.py data/videos/video_27.mp4 `
  --tracker configs/custom_botsort.yaml `
  --rules-config configs/snatch_rules.yaml
```

Chỉ chạy detection/tracking:

```powershell
uv run main.py data/videos/video_27.mp4 --disable-analytics
```

## Kiểm thử

```powershell
uv run python test.py
```

## Cấu hình và kết quả

- `configs/snatch_rules.yaml`: ngưỡng, trọng số evidence và version của rule engine.
- `configs/custom_tracker.yaml`, `configs/custom_botsort.yaml`: cấu hình tracker Ultralytics.
- `data/database/`: SQLite chứa `entity_observations`, `bag_person_relations`, `snatch_events` và `person_roles`.
- `script_rule_engine.md`: giải thích state machine và cách tuning rule.

## REST API

```powershell
uv run fastapi run src/api/app.py
```

- `GET /health`: kiểm tra API.
- `POST /analyses`: tạo job phân tích từ local `source` path.
- `GET /analyses/{job_id}`: xem trạng thái `queued`, `running`, `completed` hoặc `failed`.

API này giữ job trong bộ nhớ để codebase ngắn gọn; khi triển khai CCTV nhiều tiến trình, chuyển job state sang database hoặc message queue.
