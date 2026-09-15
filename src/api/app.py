"""REST API ngắn gọn bao quanh pipeline phân tích video."""

from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.pipeline.detection_pipeline import DetectionPipeline


app = FastAPI(title="TROMCAP Snatching Detection API", version="0.1.0")
_jobs: dict[str, dict[str, object]] = {}
_jobs_lock = Lock()


class AnalysisRequest(BaseModel):
    """Configuration for one local image or video analysis job."""

    source: str = Field(min_length=1, description="Local image or video path")
    model_path: str = "models/detection/best.pt"
    db_path: str = "data/database/detections.db"
    tracker_config: str = "configs/custom_tracker.yaml"
    rules_config: str = "configs/snatch_rules.yaml"
    output_dir: str | None = "data/output_video"
    tracking_conf: float = Field(default=0.1, ge=0.0, le=1.0)
    conf: float | None = Field(default=None, ge=0.0, le=1.0)
    person_conf: float = Field(default=0.6, ge=0.0, le=1.0)
    bag_conf: float = Field(default=0.25, ge=0.0, le=1.0)
    default_conf: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_analytics: bool = True


def _snapshot(job_id: str) -> dict[str, object]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job was not found")
        return dict(job)


def _run_job(job_id: str, request: AnalysisRequest) -> None:
    pipeline: DetectionPipeline | None = None
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
        confidence: float | dict[str, float] = (
            request.conf
            if request.conf is not None
            else {
                "person": request.person_conf,
                "bag": request.bag_conf,
                "default": request.default_conf,
            }
        )
        pipeline = DetectionPipeline(
            model_path=request.model_path,
            db_path=request.db_path,
            conf=confidence,
            tracker_config=request.tracker_config,
            tracking_conf=request.tracking_conf,
            rules_config=request.rules_config,
            enable_analytics=request.enable_analytics,
        )
        video_id = pipeline.run(request.source, request.output_dir or None)
        with _jobs_lock:
            _jobs[job_id].update(status="completed", video_id=video_id)
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=str(exc))
    finally:
        if pipeline is not None:
            pipeline.db.close()


@app.get("/health")
def health() -> dict[str, str]:
    """Return API liveness without loading an ML model."""
    return {"status": "ok"}


@app.post("/analyses", status_code=status.HTTP_202_ACCEPTED)
def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    """Queue a video or image analysis job and return its identifier."""
    if not Path(request.source).is_file():
        raise HTTPException(status_code=404, detail="Source file was not found")

    job_id = str(uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued"}
    background_tasks.add_task(_run_job, job_id, request)
    return _snapshot(job_id)


@app.get("/analyses/{job_id}")
def get_analysis(job_id: str) -> dict[str, object]:
    """Return the current state of a previously queued analysis."""
    return _snapshot(job_id)
