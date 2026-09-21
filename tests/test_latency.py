import time
import unittest

from src.utils.latency import LatencyTracker


class LatencyTrackerTests(unittest.TestCase):
    def test_latency_tracker_records_times(self) -> None:
        tracker = LatencyTracker()
        self.assertEqual(tracker.summary()["count"], 0)

        # Simulate 2 frames
        for _ in range(2):
            tracker.start_frame()
            time.sleep(0.005)
            tracker.mark_tracking_done()
            time.sleep(0.005)
            track_ms, post_ms, total_ms = tracker.end_frame()

            self.assertGreater(track_ms, 0.0)
            self.assertGreater(post_ms, 0.0)
            self.assertGreaterEqual(total_ms, track_ms + post_ms - 0.5)

        summary = tracker.summary()
        self.assertEqual(summary["count"], 2)
        self.assertGreater(summary["avg_track_ms"], 0.0)
        self.assertGreater(summary["avg_analytics_post_ms"], 0.0)
        self.assertGreater(summary["avg_total_ms"], 0.0)
        self.assertGreater(summary["fps"], 0.0)

    def test_reset_clears_metrics(self) -> None:
        tracker = LatencyTracker()
        tracker.start_frame()
        time.sleep(0.002)
        tracker.mark_tracking_done()
        time.sleep(0.002)
        tracker.end_frame()

        self.assertEqual(tracker.summary()["count"], 1)
        tracker.reset()
        self.assertEqual(tracker.summary()["count"], 0)
        self.assertEqual(tracker.summary()["fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
