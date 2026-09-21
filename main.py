"""Điểm vào dòng lệnh của TROMCAP."""

import argparse

from src.pipeline.detection_pipeline import DetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TROMCAP Snatching Detection")
    parser.add_argument("source", help="Path to a video or image")
    parser.add_argument(
        "--model",
        default="models/detection/best.pt",
        help="Path to YOLO weights",
    )
    parser.add_argument(
        "--db",
        default="data/database/detections.db",
        help="Path to the SQLite output database",
    )
    parser.add_argument(
        "--tracker",
        default="configs/custom_botsort.yaml",
        help="Path to an Ultralytics tracker YAML configuration",
    )
    parser.add_argument(
        "--track-conf",
        type=float,
        default=0.1,
        help="Minimum detector confidence passed into the tracker",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override one detector output threshold for every class",
    )
    parser.add_argument("--person-conf", type=float, default=0.6)
    parser.add_argument("--bag-conf", type=float, default=0.25)
    parser.add_argument("--default-conf", type=float, default=0.5)
    parser.add_argument(
        "--rules-config",
        default="configs/snatch_rules.yaml",
        help="Path to person-bag and snatch rule configuration",
    )
    parser.add_argument(
        "--disable-analytics",
        action="store_true",
        help="Run detection/tracking without person-bag rules",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output_video/vd",
        help="Annotated output directory; pass an empty string to disable",
    )
    parser.add_argument("--show", action="store_true", help="Show live annotated frames")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    confidence: float | dict[str, float]
    if args.conf is not None:
        confidence = args.conf
    else:
        confidence = {
            "person": args.person_conf,
            "bag": args.bag_conf,
            "default": args.default_conf,
        }

    pipeline = DetectionPipeline(
        model_path=args.model,
        db_path=args.db,
        conf=confidence,
        tracker_config=args.tracker,
        tracking_conf=args.track_conf,
        rules_config=args.rules_config,
        enable_analytics=not args.disable_analytics,
    )
    try:
        video_id = pipeline.run(
            args.source,
            output_dir=args.output_dir or None,
            show=args.show,
        )
        print(f"Completed. video_id={video_id}")
    finally:
        pipeline.db.close()


if __name__ == "__main__":
    main()
