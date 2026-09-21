"""Bộ theo dõi độ trễ đơn giản để đo hiệu năng pipeline."""

import time
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    """Measures tracking latency, analytics-to-end latency, and end-to-end latency."""

    track_times: list[float] = field(default_factory=list)
    analytics_post_times: list[float] = field(default_factory=list)
    total_times: list[float] = field(default_factory=list)

    _t_start: float = 0.0
    _t_track_end: float = 0.0

    def start_frame(self) -> None:
        """Mark start of frame processing (capture + tracking)."""
        self._t_start = time.perf_counter()

    def mark_tracking_done(self) -> None:
        """Mark completion of tracking."""
        self._t_track_end = time.perf_counter()

    def end_frame(self) -> tuple[float, float, float]:
        """Mark end of frame processing (after analytics, DB, vis, write).

        Returns:
            tuple[float, float, float]: (track_ms, analytics_post_ms, total_ms)
        """
        t_end = time.perf_counter()
        track_ms = (self._t_track_end - self._t_start) * 1000.0
        analytics_post_ms = (t_end - self._t_track_end) * 1000.0
        total_ms = (t_end - self._t_start) * 1000.0

        self.track_times.append(track_ms)
        self.analytics_post_times.append(analytics_post_ms)
        self.total_times.append(total_ms)

        return track_ms, analytics_post_ms, total_ms

    def reset(self) -> None:
        """Reset recorded measurements."""
        self.track_times.clear()
        self.analytics_post_times.clear()
        self.total_times.clear()
        self._t_start = 0.0
        self._t_track_end = 0.0

    def summary(self) -> dict[str, float]:
        """Compute average latency and FPS."""
        n = len(self.total_times)
        if n == 0:
            return {
                "count": 0,
                "avg_track_ms": 0.0,
                "avg_analytics_post_ms": 0.0,
                "avg_total_ms": 0.0,
                "fps": 0.0,
            }
        avg_track = sum(self.track_times) / n
        avg_analytics_post = sum(self.analytics_post_times) / n
        avg_total = sum(self.total_times) / n
        fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        return {
            "count": n,
            "avg_track_ms": avg_track,
            "avg_analytics_post_ms": avg_analytics_post,
            "avg_total_ms": avg_total,
            "fps": fps,
        }

    def print_summary(self) -> None:
        """Print latency summary to console."""
        s = self.summary()
        if s["count"] == 0:
            print("No frames recorded for latency measurement.")
            return
        print("\n" + "=" * 55)
        print(f"Latency Summary (over {int(s['count'])} frames):")
        print(f"  - Tu dau -> Tracking xong : {s['avg_track_ms']:.2f} ms")
        print(f"  - Analytics -> Den het   : {s['avg_analytics_post_ms']:.2f} ms")
        print(f"  - Tong End-to-End        : {s['avg_total_ms']:.2f} ms")
        print(f"  - Estimated Average FPS  : {s['fps']:.2f} FPS")
        print("=" * 55 + "\n")
