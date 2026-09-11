"""FastAPI service for the SIH SatQuery working demo.

M6-facing API over the production M4 controller.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    version="0.5.1",
)

# ---------------------------------------------------------------
# Development/frontend CORS support.
# ---------------------------------------------------------------
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "SATQUERY_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)

# One production M4 API instance for the service.
satquery = SatQueryAPI()


class MissionRequest(BaseModel):
    """JSON request accepted by the stable M4 mission endpoint."""

    query: str = Field(min_length=1)
    context: dict = Field(default_factory=dict)


def _dump_model(value):
    """Serialize a Pydantic model with v2/v1 compatibility."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _artifact_urls(result: dict) -> list[str]:
    """Convert returned artifact paths into browser URLs."""
    urls = []

    for item in result.get("results", []):
        data = item.get("data", {}) or {}

        for artifact in data.get("artifacts", []) or []:
            name = Path(str(artifact)).name
            urls.append(f"/outputs/{name}")

        for artifact in data.get("artifact_paths", []) or []:
            name = Path(str(artifact)).name
            urls.append(f"/outputs/{name}")

        for key in (
            "evidence_path",
            "geojson_path",
            "map_path",
        ):
            artifact = data.get(key)
            if artifact:
                name = Path(str(artifact)).name
                urls.append(f"/outputs/{name}")

        for artifact in item.get("evidence", []) or []:
            reference = artifact.get("reference")
            if reference:
                path = Path(str(reference))
                if path.exists() and path.is_file():
                    urls.append(f"/outputs/{path.name}")

    return list(dict.fromkeys(urls))


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    """Persist one uploaded file using a sanitized filename."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    return destination


def _run_live_mission(
    *,
    query: str,
    mode: str,
    image: UploadFile | None = None,
    before_image: UploadFile | None = None,
    after_image: UploadFile | None = None,
    optical_image: UploadFile | None = None,
    sar_image: UploadFile | None = None,
) -> dict:
    """Accept the M6 multipart contract and invoke the real M4 pipeline."""

    normalized_mode = mode.strip().lower()
    allowed_modes = {
        "image-understanding",
        "change-detection",
        "optical-sar",
    }

    if normalized_mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported analysis mode '{mode}'. "
                f"Expected one of: {', '.join(sorted(allowed_modes))}."
            ),
        )

    mission_id = uuid4().hex[:12]
    mission_dir = RUNTIME_DIR / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)

    context: dict[str, object] = {
        "output_dir": str(OUTPUT_DIR),
    }

    saved_files: dict[str, Path] = {}

    if normalized_mode == "image-understanding":
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Image-understanding mode requires an 'image' upload.",
            )

        path = _save_upload(
            image,
            mission_dir / f"primary-{Path(image.filename or 'image').name}",
        )
        saved_files["primary"] = path
        context["images"] = [str(path)]

    elif normalized_mode == "change-detection":
        if before_image is None or after_image is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Change-detection mode requires both "
                    "'before_image' and 'after_image' uploads."
                ),
            )

        before_path = _save_upload(
            before_image,
            mission_dir / f"before-{Path(before_image.filename or 'image').name}",
        )
        after_path = _save_upload(
            after_image,
            mission_dir / f"after-{Path(after_image.filename or 'image').name}",
        )
        saved_files["before"] = before_path
        saved_files["after"] = after_path
        context.update(
            {
                "before": str(before_path),
                "after": str(after_path),
            }
        )

    else:  # optical-sar
        if optical_image is None or sar_image is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Optical-SAR mode requires both "
                    "'optical_image' and 'sar_image' uploads."
                ),
            )

        optical_path = _save_upload(
            optical_image,
            mission_dir / f"optical-{Path(optical_image.filename or 'image').name}",
        )
        sar_path = _save_upload(
            sar_image,
            mission_dir / f"sar-{Path(sar_image.filename or 'image').name}",
        )
        saved_files["optical"] = optical_path
        saved_files["sar"] = sar_path
        context.update(
            {
                "optical": str(optical_path),
                "sar": str(sar_path),
            }
        )

    started = time.perf_counter()

    try:
        response = satquery.ask(
            query,
            context=context,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SatQuery execution failed: {exc}",
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result = _dump_model(response)

    result["mission_id"] = mission_id
    result["execution_time_ms"] = elapsed_ms
    result["artifacts"] = _artifact_urls(result)
    result["uploaded_files"] = {
        key: str(path.relative_to(RUNTIME_DIR))
        for key, path in saved_files.items()
    }

    return result


@app.get("/")
def workspace() -> FileResponse:
    """Serve the simple M6 judge-facing frontend."""
    return FileResponse(BASE_DIR / "static" / "m6.html")


@app.get("/m6")
def m6_frontend() -> FileResponse:
    """Explicit route for the simple M6 mission frontend."""
    return FileResponse(BASE_DIR / "static" / "m6.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    """Explicit route for the detailed developer dashboard."""
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/health")
def health() -> dict:
    """Service health endpoint."""
    return {
        "status": "ok",
        "service": "satquery-ai",
        "version": app.version,
        "architecture": "M6 -> M5 API -> M4 -> M1/M2/M3/M5-GIS",
    }


@app.get("/api/health")
def api_health() -> dict:
    """Compatibility alias for the M6 frontend health check."""
    return health()


@app.post("/api/v1/mission")
def mission(request: MissionRequest) -> dict:
    """Run a natural-language mission through production M4."""
    response = satquery.ask(
        request.query,
        context=request.context,
    )

    result = _dump_model(response)
    result["mission_id"] = uuid4().hex[:12]
    result["execution_time_ms"] = 0
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
    """Run an uploaded temporal pair through production M4."""

    return _run_live_mission(
        query=query,
        mode="change-detection",
        before_image=before,
        after_image=after,
    )


@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    mode: str = Form(...),
    image: UploadFile | None = File(default=None),
    before_image: UploadFile | None = File(default=None),
    after_image: UploadFile | None = File(default=None),
    optical_image: UploadFile | None = File(default=None),
    sar_image: UploadFile | None = File(default=None),
) -> dict:
    """M6 multipart endpoint for live analysis."""

    return _run_live_mission(
        query=query,
        mode=mode,
        image=image,
        before_image=before_image,
        after_image=after_image,
        optical_image=optical_image,
        sar_image=sar_image,
    )
