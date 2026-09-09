"""FastAPI service for the SIH SatQuery working demo."""
from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mission.orchestrator import run_mission


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SatQuery AI", version="0.4.0")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


class MissionRequest(BaseModel):
    query: str = Field(min_length=1)
    context: dict = Field(default_factory=dict)


@app.get("/")
def workspace() -> FileResponse:
    """Serve the judge-facing mission workspace."""
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    """Explicit route for the SIH judge/demo dashboard."""
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/health")
def health() -> dict:
    """Service health endpoint for deployment checks."""
    return {"status": "ok", "service": "satquery-ai", "version": app.version}


@app.post("/api/v1/mission")
def mission(request: MissionRequest) -> dict:
    """Run an orchestration mission using already-available local paths."""
    return run_mission(request.query, request.context)


@app.post("/api/v1/change-detection")
async def change_detection_demo(
    query: str = Form("Find changes between the two images"),
    before: UploadFile = File(...),
    after: UploadFile = File(...),
) -> dict:
    """Persist uploads, execute M2, and return browser-viewable artifacts."""
    mission_id = uuid4().hex[:12]
    mission_dir = RUNTIME_DIR / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)

    before_path = mission_dir / f"before-{Path(before.filename or 'image').name}"
    after_path = mission_dir / f"after-{Path(after.filename or 'image').name}"
    with before_path.open("wb") as handle:
        shutil.copyfileobj(before.file, handle)
    with after_path.open("wb") as handle:
        shutil.copyfileobj(after.file, handle)

    result = run_mission(
        query,
        {
            "before": str(before_path),
            "after": str(after_path),
            "output_dir": str(OUTPUT_DIR),
        },
    )

    for item in result["results"]:
        item["artifact_urls"] = [f"/outputs/{name}" for name in item.get("artifacts", [])]

    result["mission_id"] = mission_id
    result["artifacts"] = [
        url
        for item in result["results"]
        for url in item.get("artifact_urls", [])
    ]
    return result
