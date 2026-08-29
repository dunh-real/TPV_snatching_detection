"""Tracker profiles and qualitative guidance used by the benchmark."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackerProfile:
    """One tracker configuration plus project-specific qualitative notes."""

    name: str
    algorithm: str
    config: str
    reid: bool
    strengths: str
    weaknesses: str
    fit_for_project: str

    def resolved_config(self, repo_root: Path) -> str:
        """Resolve repository configs while leaving Ultralytics built-ins untouched."""
        candidate = repo_root / self.config
        return str(candidate.resolve()) if candidate.is_file() else self.config


PROFILES: dict[str, TrackerProfile] = {
    "bytetrack_tuned": TrackerProfile(
        name="bytetrack_tuned",
        algorithm="ByteTrack",
        config="configs/custom_tracker.yaml",
        reid=False,
        strengths="Nhanh, đơn giản, tận dụng detection confidence thấp; hợp camera CCTV tĩnh.",
        weaknesses="Không có appearance/ReID; dễ vỡ ID khi box biến dạng mạnh hoặc mất lâu.",
        fit_for_project="Baseline chính cho edge; ưu tiên nếu detector bag có recall tốt.",
    ),
    "botsort_reid_tuned": TrackerProfile(
        name="botsort_reid_tuned",
        algorithm="BoT-SORT",
        config="configs/custom_botsort.yaml",
        reid=True,
        strengths="Kết hợp chuyển động và appearance; có thể nối track qua che khuất.",
        weaknesses="Tốn tài nguyên hơn; ReID tổng quát có thể nối nhầm người/túi giống nhau.",
        fit_for_project="Ứng viên khi ID switch người là lỗi chính; phải kiểm tra riêng class bag.",
    ),
    "ocsort_default": TrackerProfile(
        name="ocsort_default",
        algorithm="OC-SORT",
        config="ocsort.yaml",
        reid=False,
        strengths="Observation-centric; xử lý chuyển động đổi hướng tốt hơn Kalman thuần.",
        weaknesses="Không appearance; config mặc định chưa tối ưu cho confidence của model bag.",
        fit_for_project="Nên thử với các pha chạy xe/rời hiện trường có chuyển động đột ngột.",
    ),
    "deepocsort_default": TrackerProfile(
        name="deepocsort_default",
        algorithm="Deep OC-SORT",
        config="deepocsort.yaml",
        reid=False,
        strengths="Mô hình chuyển động observation-centric và hỗ trợ ReID khi bật.",
        weaknesses="Profile mặc định đang tắt ReID; nhiều tham số và nặng hơn OC-SORT.",
        fit_for_project="Dùng làm mốc OC-SORT nâng cao trước khi tạo profile ReID riêng.",
    ),
    "fasttrack_default": TrackerProfile(
        name="fasttrack_default",
        algorithm="FastTrack",
        config="fasttrack.yaml",
        reid=False,
        strengths="Có rollback Kalman và logic che khuất, không cần encoder appearance.",
        weaknesses="Phụ thuộc các heuristic che khuất và ngưỡng bbox; cần tune theo góc CCTV.",
        fit_for_project="Đáng thử cho edge khi cần chịu che khuất nhưng không đủ tài nguyên ReID.",
    ),
    "tracktrack_default": TrackerProfile(
        name="tracktrack_default",
        algorithm="TrackTrack",
        config="tracktrack.yaml",
        reid=False,
        strengths="Ghép nhiều tín hiệu, iterative assignment và hạn chế tạo duplicate track.",
        weaknesses="Nhiều hyperparameter; ngưỡng mặc định cao có thể bỏ sót bag confidence thấp.",
        fit_for_project="Ứng viên nghiên cứu; cần tune confidence trước khi kết luận.",
    ),
    "bytetrack_default": TrackerProfile(
        name="bytetrack_default",
        algorithm="ByteTrack",
        config="bytetrack.yaml",
        reid=False,
        strengths="Mốc mặc định của Ultralytics, dễ tái lập.",
        weaknesses="Tạo track mới ở confidence thấp hơn profile tuned, dễ sinh false track bag.",
        fit_for_project="Dùng để đo lợi ích thực sự của custom_tracker.yaml.",
    ),
    "botsort_default": TrackerProfile(
        name="botsort_default",
        algorithm="BoT-SORT",
        config="botsort.yaml",
        reid=False,
        strengths="Có camera-motion compensation; mốc mặc định dễ tái lập.",
        weaknesses="ReID mặc định tắt; GMC không cần thiết cho camera CCTV hoàn toàn tĩnh.",
        fit_for_project="Dùng để tách lợi ích của ReID/tuning khỏi bản thân BoT-SORT.",
    ),
}


DEFAULT_PROFILE_NAMES = (
    "bytetrack_tuned",
    "botsort_reid_tuned",
    "ocsort_default",
    "deepocsort_default",
    "fasttrack_default",
    "tracktrack_default",
)


def select_profiles(names: list[str] | None) -> list[TrackerProfile]:
    """Return requested profiles, raising a clear error for unknown names."""
    selected_names = names or list(DEFAULT_PROFILE_NAMES)
    unknown = sorted(set(selected_names) - PROFILES.keys())
    if unknown:
        raise ValueError(f"Unknown tracker profiles: {', '.join(unknown)}")
    return [PROFILES[name] for name in selected_names]
