import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_concurrent import (
    _resource_summary,
    expected_frames,
    find_videos,
    percentile,
)


class ConcurrentBenchmarkTests(unittest.TestCase):
    def test_expected_frames_excludes_end_boundary(self) -> None:
        self.assertEqual(expected_frames(2.0, 30.0), 60)
        self.assertEqual(expected_frames(1.05, 10.0), 11)

    def test_percentile(self) -> None:
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([10, 20, 30], 0.5), 20)
        self.assertEqual(percentile([10, 20, 30], 0.95), 29)

    def test_resource_summary_ignores_missing_gpu_samples(self) -> None:
        result = _resource_summary([
            {"rss_mb": 100, "cpu_cores": None, "gpu_util_percent": None, "gpu_vram_mb": None},
            {"rss_mb": 120, "cpu_cores": 1.5, "gpu_util_percent": 80, "gpu_vram_mb": 1000},
        ], 400)
        self.assertEqual(result["worker_rss_peak_mb"], 120)
        self.assertEqual(result["gpu_vram_peak_delta_mb"], 600)
        self.assertEqual(result["cpu_avg_cores"], 1.5)

    def test_find_videos_filters_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.mp4").touch()
            (root / "b.txt").touch()
            self.assertEqual(find_videos(root), [root / "a.mp4"])


if __name__ == "__main__":
    unittest.main()
