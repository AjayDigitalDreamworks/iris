"""Video streaming endpoints.

Video never travels over the WebSocket event channel.  These are separate,
authorised HTTP endpoints that stream JPEG frames from the camera's newest
buffered frame (capture is decoupled from inference).
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.errors import NotFoundError, SourceUnavailableError
from app.core.logging import get_logger
from app.models import Camera
from app.services.camera_manager import (
    MJPEG_BOUNDARY,
    build_multipart_frame,
    camera_manager,
    encode_jpeg,
)

router = APIRouter(prefix="/streams", tags=["streams"])
logger = get_logger("camera")

# The boundary used both in the response Content-Type and in the part
# delimiters. Sourced from one constant so the two can never drift apart again.
BOUNDARY = MJPEG_BOUNDARY


async def _mjpeg_stream(
    camera_id: str,
    fps: float,
    camera_name: str = "",
    annotated: bool = True,
    persons_only: bool = False,
):
    """Yield multipart MJPEG chunks from the camera's latest-frame buffer."""
    worker = camera_manager.worker(camera_id)
    target_fps = max(0.5, min(fps, 30.0))
    interval = 1.0 / target_fps
    last_frame_key: tuple[int, int] | None = None
    idle_sent = 0
    try:
        while True:
            frame = await asyncio.to_thread(
                worker.latest_frame, timeout=0.5, annotated=annotated, persons_only=persons_only
            )
            if frame is None:
                idle_sent += 1
                if idle_sent > 60:
                    # The camera is not producing frames: close rather than spin.
                    logger.info(
                        "Closing MJPEG stream: no frames",
                        extra={"camera_id": camera_id},
                    )
                    return
                await asyncio.sleep(interval)
                continue
            idle_sent = 0
            det_count = len(worker.runner.live_detections) if (worker and worker.runner) else 0
            current_key = (frame.index, det_count if (annotated or persons_only) else 0)
            if current_key != last_frame_key:
                last_frame_key = current_key
                jpeg = await asyncio.to_thread(encode_jpeg, frame)
                yield build_multipart_frame(jpeg)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:  # client disconnected
        raise
    finally:
        logger.info("MJPEG stream closed", extra={"camera_id": camera_id})


@router.get(
    "/{camera_id}/mjpeg",
    summary="Live MJPEG stream for a camera (authorised)",
    response_class=StreamingResponse,
    responses={
        # FastAPI cannot infer the media type from a StreamingResponse whose
        # content type carries a boundary parameter, so state it explicitly.
        200: {
            "content": {"multipart/x-mixed-replace": {}},
            "description": "Endless multipart stream of JPEG frames.",
        },
        404: {"description": "Unknown camera."},
    },
)
async def stream_mjpeg(
    camera_id: str,
    session: DbSession,
    _: CurrentUser,
    fps: float = Query(default=25.0, ge=0.5, le=30.0),
    annotated: bool = Query(default=True),
    layer: str = Query(default="all", pattern="^(all|persons)$"),
) -> StreamingResponse:
    # Resolve the camera, then RELEASE the request session before streaming:
    # a long-lived stream must never hold a database connection.
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera {camera_id!r} does not exist.")
    name = camera.name
    session.close()
    return StreamingResponse(
        _mjpeg_stream(
            camera_id,
            fps,
            name,
            annotated=annotated,
            persons_only=(layer == "persons"),
        ),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "X-Accel-Buffering": "no",
            "Pragma": "no-cache",
            "X-Camera-Name": name.encode("ascii", "ignore").decode() or camera_id,
        },
    )


@router.get(
    "/{camera_id}/snapshot.jpg",
    summary="Single JPEG snapshot from a camera (authorised)",
    # Declared as a plain Response so OpenAPI generation has a response class
    # to describe; the body is a raw JPEG, not JSON.
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Newest buffered frame."},
        404: {"description": "Unknown camera."},
        503: {"description": "Camera not running, or no frame buffered yet."},
    },
)
async def snapshot(camera_id: str, session: DbSession, _: CurrentUser):
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera {camera_id!r} does not exist.")
    worker = camera_manager.workers.get(camera_id)
    if worker is None:
        raise SourceUnavailableError(
            "This camera is not running. Start it before requesting a snapshot."
        )
    session.close()
    frame = await asyncio.to_thread(worker.latest_frame, timeout=2.0)
    if frame is None:
        raise SourceUnavailableError("No frame is available yet from this camera.")
    jpeg = await asyncio.to_thread(encode_jpeg, frame)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Frame-Timestamp": f"{time.time():.3f}"},
    )


@router.get("/{camera_id}/status", summary="Streaming and pipeline status")
def stream_status(camera_id: str, session: DbSession, _: CurrentUser) -> dict:
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera {camera_id!r} does not exist.")
    worker = camera_manager.workers.get(camera_id)
    health = camera_manager.health(camera_id)
    session.close()
    return {
        "camera_id": camera_id,
        "camera_name": camera.name,
        "running": bool(worker and worker.is_running),
        "mjpeg_url": f"/api/v1/streams/{camera_id}/mjpeg",
        "snapshot_url": f"/api/v1/streams/{camera_id}/snapshot.jpg",
        "max_clients": settings.mjpeg_max_clients,
        "queue_depth": health["queue_depth"],
        "frames_processed": health["frames_processed"],
        "mjpeg_boundary": BOUNDARY,
    }


@router.get(
    "/{camera_id}/detections",
    summary="Live detections for the newest analysed frame",
)
def live_detections(camera_id: str, session: DbSession, _: CurrentUser) -> dict:
    """Boxes for the CURRENT frame, from memory.

    The Live overlay must track what is on screen now. Replaying stored
    detection history draws many frames' boxes at once and makes the model look
    like it over-detects; this endpoint returns only the latest frame's set.
    """
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera {camera_id!r} does not exist.")
    worker = camera_manager.workers.get(camera_id)
    session.close()
    if worker is None:
        return {"camera_id": camera_id, "detections": [], "frame": {}}
    return worker.latest_detections()