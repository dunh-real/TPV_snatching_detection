"""Small dependency-free tests for benchmark continuity metrics."""

import unittest

from experiments.tracker_benchmark.metrics import (
    TrackPoint,
    find_fragment_candidates,
    split_tracks,
    summarize_video,
)


def point(frame: int, track_id: int, x: float, label: str = "person") -> TrackPoint:
    return TrackPoint(
        frame_idx=frame,
        timestamp_ms=frame * 100.0,
        track_id=track_id,
        label=label,
        class_id=0 if label == "person" else 1,
        confidence=0.8,
        bbox=(x, 0.0, x + 20.0, 40.0),
    )


class TrackerMetricTests(unittest.TestCase):
    def test_fragment_candidate_links_close_tracklets(self) -> None:
        points = [
            point(0, 1, 0.0),
            point(1, 1, 2.0),
            point(3, 2, 6.0),
            point(4, 2, 8.0),
            point(3, 3, 500.0),
        ]
        candidates = find_fragment_candidates(
            "tracker", "video.mp4", split_tracks(points), max_gap=5
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["old_track_id"], 1)
        self.assertEqual(candidates[0]["new_track_id"], 2)

    def test_summary_counts_track_gaps_and_coverage(self) -> None:
        points = [point(0, 1, 0.0), point(2, 1, 2.0), point(1, 2, 50.0, "bag")]
        summary = summarize_video(
            tracker_name="tracker",
            video_name="video.mp4",
            points=points,
            decoded_frames=4,
            metadata_frames=4,
            source_fps=10.0,
            tracker_seconds=1.0,
            first_frame_seconds=0.1,
            wall_seconds=1.2,
            fragment_count=0,
        )
        self.assertEqual(summary["unique_tracks"], 2)
        self.assertEqual(summary["internal_gap_frames"], 1)
        self.assertEqual(summary["frames_with_any_track"], 3)
        self.assertAlmostEqual(summary["coverage_any"], 0.75)


if __name__ == "__main__":
    unittest.main()
