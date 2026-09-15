# Hướng dẫn cấu trúc mã nguồn

Các thư mục cấp cao dưới `src/` được giữ cố định: `api`, `core`, `database`, `pipeline`, `services` và `utils`.

```text
src/
├── api/
│   └── __init__.py                  # Chỉ là điểm mở rộng HTTP API, chưa có endpoint
├── core/
│   ├── types.py                     # Hợp đồng Detection/TrackedObject dùng chung
│   ├── vision/
│   │   ├── detector.py              # Ultralytics YOLO adapter
│   │   └── tracker.py               # ByteTrack/BoT-SORT adapter
│   └── analytics/
│       ├── config.py                # Dataclass config và YAML loader
│       ├── types.py                 # Entity, motion, relation, event, role
│       ├── geometry.py              # Hình học phục vụ rule engine
│       ├── identity.py              # Canonical entity identity
│       ├── trajectory.py            # Smoothing/kinematics
│       ├── person_bag.py            # Holder association state machine
│       ├── snatch_detector.py       # Snatch event state machine
│       └── roles.py                 # Victim/suspect role resolution
├── database/
│   ├── connection.py                # Tạo SQLite connection
│   ├── schema.py                    # Schema và index
│   ├── detection_repository.py      # Video và detection persistence
│   ├── analytics_repository.py      # Analytics persistence
│   └── operations.py                # DetectionDB compatibility facade
├── pipeline/
│   └── detection_pipeline.py        # Điều phối input → output
├── services/
│   ├── object_detect.py             # Detection/tracking application service
│   └── snatch_analytics.py          # Rule-engine application service
└── utils/
    ├── latency.py                   # Performance metrics
    └── visualizer.py                # OpenCV output rendering
```

## Quy tắc phụ thuộc

- `core/analytics` không import OpenCV, Ultralytics hoặc SQLite; nó chỉ dùng các dependency cấu hình cần thiết như PyYAML.
- `core/vision` sở hữu dependency Ultralytics; không import `services`, `pipeline` hay `database`.
- `services` kết hợp `core` thành API nghiệp vụ. `DetectionPipeline` sử dụng service thay vì tự khởi tạo detector/tracker.
- `database` chỉ nhận các data model làm input/output; không điều khiển pipeline hay rule engine.
- `utils` không được trở thành nơi chứa logic nghiệp vụ.

Mọi thay đổi thuật toán, threshold và SQLite schema phải có test tương ứng. Việc tổ chức lại module không được thay đổi API công khai của `DetectionDB`, `ObjectDetectionService` hoặc `SnatchAnalyticsEngine`.
