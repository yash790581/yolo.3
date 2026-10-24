"""
UrbanSense AI - Centralized Backend Server (MVC Architecture)
FastAPI Application coordinating Edge AI telemetry, MongoDB Atlas persistence,
and GIS dashboard APIs.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import (
    MEDIA_DIR,
    SAMPLE_DIR,
    UPLOADS_DIR,
    DATA_DIR,
    HOST,
    PORT,
    DEBUG
)
from backend.routes import (
    health_router,
    detection_router,
    event_router,
    bus_router,
    route_router,
    analytics_router,
    demo_router,
    video_analysis_router,
    admin_router,
    database_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Initializing UrbanSense AI Platform (Clean Zero-Data Mode)...")
    print(f"[Server] Platform ready and listening on http://{HOST}:{PORT}")
    yield


app = FastAPI(
    title="UrbanSense AI - Centralized Urban Intelligence Platform",
    description="Transforms public transit buses into mobile sensing units using onboard YOLO edge AI.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directories for media serving
app.mount("/outputs", StaticFiles(directory=str(MEDIA_DIR)), name="outputs")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/sample_data", StaticFiles(directory=str(SAMPLE_DIR)), name="sample_data")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Register RESTful MVC routers
app.include_router(health_router)
app.include_router(detection_router)
app.include_router(event_router)
app.include_router(bus_router)
app.include_router(route_router)
app.include_router(analytics_router)
app.include_router(demo_router)
app.include_router(video_analysis_router)
app.include_router(admin_router)
app.include_router(database_router)

# ── Serve Built Frontend (SPA) if dist/ exists ────────────────────────────────
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Do not catch unresolved API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"API route '/{full_path}' not found")
        target_file = FRONTEND_DIST / full_path
        if full_path and target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=DEBUG)
