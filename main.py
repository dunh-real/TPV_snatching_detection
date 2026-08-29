"""TROMCAP — Traffic Object Recognition and Motion Capture."""

import argparse
from pathlib import Path

from src.pipeline.detection_pipeline import DetectionPipeline

custom_conf = {
    "person": 0.5,
    "bag": 0.25,
    "default": 0.4
}
out = Path("data/output_video")

def main():
    parser = argparse.ArgumentParser(description="TROMCAP Snatching Detection")
    parser.add_argument("source", help="Path to video or image file")
    parser.add_argument("--model", default="models/detection/best_yolov8s.pt",help="Path to YOLO weights ")
    parser.add_argument("--db", default="data/database/detections.db", help="Path to SQLite database")
    parser.add_argument("--conf", type=float, default=custom_conf, help="Detection confidence threshold")
    parser.add_argument(
        "--tracker",
        default="configs/custom_tracker.yaml",
        help="Path to an Ultralytics tracker YAML configuration",
    )
    parser.add_argument(
        "--track-conf",
        type=float,
        default=0.1,
        help="Minimum detector confidence passed into the tracker",
    )
    parser.add_argument("--output-dir", "-o", default="data/output_video", help="Directory to save output video/image")
    parser.add_argument("--show", action="store_true", help="Display results in a window")
    args = parser.parse_args()


    pipeline = DetectionPipeline(
        model_path=args.model,
        db_path=args.db,
        conf=args.conf,
        tracker_config=args.tracker,
        tracking_conf=args.track_conf,
    )
    try:
        video_id = pipeline.run(args.source, args.output_dir, args.show)
        print(f"Completed. video_id={video_id}")
    finally:
        pipeline.db.close()


if __name__ == "__main__":
    main()
