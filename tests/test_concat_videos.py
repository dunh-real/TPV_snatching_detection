"""Check time-preserving concatenation and boundary metadata."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.concat_videos import concat_videos


def write_clip(path: Path, fps: float, size: tuple[int, int], frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Cannot create test video")
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), 120, dtype=np.uint8))
    writer.release()


class ConcatVideosTest(unittest.TestCase):
    def test_mixed_fps_and_sizes_keep_duration_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            output = root / "joined.mp4"
            manifest = root / "joined.segments.json"
            write_clip(first, fps=10, size=(64, 48), frames=10)
            write_clip(second, fps=20, size=(48, 64), frames=20)

            result = concat_videos(
                [first, second], output, manifest, target_fps=10,
                dimensions=(64, 48),
            )

            self.assertEqual(result["total_frames"], 20)
            self.assertEqual(result["cut_frames"], [10])
            self.assertEqual(result["duration_s"], 2.0)
            self.assertEqual(result["segments"][1]["start_time_s"], 1.0)
            self.assertTrue(manifest.is_file())

            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 48)
            finally:
                capture.release()

    def test_refuses_to_overwrite_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            write_clip(source, fps=10, size=(64, 48), frames=2)
            with self.assertRaises(ValueError):
                concat_videos([source], source, source.with_suffix(".json"), None, None)


if __name__ == "__main__":
    unittest.main()
