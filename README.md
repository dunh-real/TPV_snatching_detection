1. Cấu trúc thư mục

tromcap/
    ├── data/                       # Quản lý dữ liệu (Không đưa lên Git)
    │   ├── raw/                    # Video gốc từ CCTV, ảnh chưa xử lý
    │   ├── processed/              # Video đã cắt frame, đã qua tiền xử lý
    │   ├── annotations/            # File nhãn (YOLO format, COCO, XML...)
    │           ├── detection/      # Nhãn cho người, xe máy, vật dụng
    │           └── action/         # Nhãn cho hành vi (snatch, run, fight)
    │   └── samples/                # Các mẫu nhỏ để test nhanh code
    ├── models/                     # Quản lý các file trọng số (.pt, .onnx, .engine)
    │   ├── detection/              # Weights của YOLOv8/v10/v11...
    │   ├── tracking/               # Config cho ByteTrack/DeepSORT
    │   └── action_recognition/     # Weights cho SlowFast/Video Swin Transformer
    ├── configs/                    # Cấu hình toàn bộ hệ thống (Cực kỳ quan trọng)
    │   ├── detection_config.yaml   # Threshold, class names, input size
    │   ├── tracking_config.yaml    # Max objects, buffer size
    │   └── camera_config.yaml      # RTSP link, góc quay, tọa độ vùng quan tâm (ROI)
    ├── src/                        # Mã nguồn chính (The Core Engine)
    │   ├── core/                   # Logic xử lý nền tảng
    │   │   ├── detection.py        # Wrapper cho các model Detection
    │   │   ├── tracking.py         # Module thực hiện Multi-Object Tracking
    │   │   └── action_recognition.py # Module phân tích hành vi theo chuỗi frame
    │   ├── pipeline/               # Quy trình chạy luồng (The Pipeline)
    │   │   ├── stream_handler.py   # Đọc luồng RTSP, xử lý drop frame, buffering
    │   │   ├── inference_engine.py # Kết hợp Detection -> Tracking -> Action
    │   │   └── post_processing.py  # Vẽ bounding box, vẽ đường track, vẽ vùng ROI
    │   ├── utils/                  # Các công cụ bổ trợ
    │   │   ├── visualization.py    # Vẽ annotation lên video để debug
    │   │   ├── geometry.py         # Tính toán khoảng cách, va chạm, giao cắt (IOU)
    │   │   └── logger.py           # Ghi log hệ thống và các cảnh báo (alerts)
    │   ├── api/                    # Interface để bên thứ 3 nhận dữ liệu
    │   │   └── alert_service.py    # Gửi thông báo qua Webhook, Telegram, MQTT
    │   └── database/               # Lưu trữ lịch sử sự kiện
    │       └── event_logger.py     # Lưu metadata về các vụ việc phát hiện được
    ├── scripts/                    # Các script hỗ trợ vận hành
    │   ├── preprocess_data.py      # Script convert format nhãn hoặc cắt video
    │   ├── train_detection.py      # Script kích hoạt quá trình training model
    │   └── export_onnx.py          # Chuyển đổi model sang định dạng tối ưu (TensorRT)
    ├── deployment/                 # Cấu hình triển khai thực tế
    │   ├── docker/                 # Dockerfile cho Edge device hoặc Server
    │   ├── edge_config/            # Config riêng cho Jetson Nano/Xavier/Orin
    │   └── cloud_config/           # Config cho triển khai trên Cloud (AWS/Azure)
    ├── tests/                      # Unit test và Integration test
    ├── docs/                       # Tài liệu kỹ thuật, sơ đồ kiến trúc
    ├── pyproject.toml              # Quản lý dependencies (uv hoặc poetry)
    ├── README.md                   # Hướng dẫn cài đặt và sử dụng
    └── .env                        # Biến môi trường (API keys, đường dẫn folder)