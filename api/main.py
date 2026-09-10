"""FastAPI service for the SIH SatQuery working demo.

M6-facing API over the production M4 controller.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api import SatQueryAPI


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
OUTPUT_DIR = RUNTIME_DIR / "outputs"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SatQuery AI",
    version="0.5.0",
)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)

# One production M4 API instance for the service.
satquery = SatQueryAPI()


class MissionRequest(BaseModel):
    query: str = Field(min_length=1)
    context: dict = Field(default_factory=dict)


def _dump_model(value):
    """Pydantic v2/v1 compatible serialization."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _artifact_urls(result: dict) -> list[str]:
    """Convert returned artifact paths into browser URLs."""
    urls = []

    for item in result.get("results", []):
        for artifact in item.get("data", {}).get("artifacts", []) or []:
            name = Path(str(artifact)).name
            urls.append(f"/outputs/{name}")

        for artifact in item.get("data", {}).get("artifact_paths", []) or []:
            name = Path(str(artifact)).name
            urls.append(f"/outputs/{name}")

        for artifact in item.get("evidence", []) or []:
            reference = artifact.get("reference")
            if reference:
                path = Path(str(reference))
                if path.exists() and path.is_file():
                    urls.append(f"/outputs/{path.name}")

    # Deduplicate while preserving order.
    return list(dict.fromkeys(urls))


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
    """Service health endpoint."""
    return {
        "status": "ok",
        "service": "satquery-ai",
        "version": app.version,
        "architecture": "M6 -> M4 -> M1/M2/M3/M5",
    }


@app.post("/api/v1/mission")
def mission(request: MissionRequest) -> dict:
    """Run a natural-language mission through production M4."""
    response = satquery.ask(
        request.query,
        context=request.context,
    )

    result = _dump_model(response)

    result["mission_id"] = uuid4().hex[:12]
    result["artifacts"] = _artifact_urls(result)

    return result


@app.post("/api/v1/change-detection")
async def change_detection_demo(
    query: str = Form(
        "Find changes between the two images"
    ),
    before: UploadFile = File(...),
    after: UploadFile = File(...),
) -> dict:
    """Run the uploaded temporal pair through production M4."""

    mission_id = uuid4().hex[:12]
    mission_dir = RUNTIME_DIR / mission_id
    mission_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_path = (
        mission_dir
        / f"before-{Path(before.filename or 'image').name}"
    )

    after_path = (
        mission_dir
        / f"after-{Path(after.filename or 'image').name}"
    )

    with before_path.open("wb") as handle:
        shutil.copyfileobj(before.file, handle)

    with after_path.open("wb") as handle:
        shutil.copyfileobj(after.file, handle)

    response = satquery.ask(
        query,
        context={
            "before": str(before_path),
            "after": str(after_path),
            "output_dir": str(OUTPUT_DIR),
        },
    )

    result = _dump_model(response)

    result["mission_id"] = mission_id
    result["artifacts"] = _artifact_urls(result)

    return result
