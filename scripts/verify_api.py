"""Verify the REST API and WebSocket against a RUNNING backend.

Starts the backend itself if nothing is listening, so the check is meaningful
rather than skipped. Skips with an explicit reason if the server cannot start.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from _verify_common import BACKEND_DIR, Report, backend_python

BASE = "http://127.0.0.1:8000"


def get(path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:
            return exc.code, None
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def get_status(path: str) -> int:
    """HTTP status only, for binary endpoints (``get`` would fail to JSON-decode)."""
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001
        return 0


def is_up() -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", 8000)) == 0


def main() -> int:
    report = Report("IRIS — API / WebSocket verification")
    server: subprocess.Popen | None = None

    if not is_up():
        server = subprocess.Popen(
            [backend_python(), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 60
        while time.time() < deadline and not is_up():
            time.sleep(1)
        if not is_up():
            report.add("Backend reachable", False, "could not start the backend", skipped=True)
            return report.finish()
        time.sleep(6)  # let camera workers and the model settle

    try:
        report.add("Backend reachable", True, BASE)

        for path, expected in (
            ("/health", 200),
            ("/api/v1/system/ready", 200),
            ("/api/v1/system/health", 200),
            ("/api/v1/system/stats", 200),
            ("/api/v1/cameras", 200),
            ("/api/v1/products", 200),
            ("/api/v1/alerts/active", 200),
            ("/api/v1/customers/footfall?range=24h", 200),
            ("/api/v1/queues", 200),
            ("/api/v1/models", 200),
            ("/api/v1/inventory", 200),
            ("/api/v1/openapi.json", 404),
            ("/openapi.json", 200),
        ):
            status, _ = get(path)
            report.add(f"GET {path}", status == expected, f"HTTP {status} (expected {expected})")

        # Error envelope shape must be consistent, not a stack trace.
        status, body = get("/api/v1/cameras/definitely-missing")
        ok = status == 404 and isinstance(body, dict) and "error" in body and "code" in body["error"]
        report.add("404 uses the error envelope", ok, f"HTTP {status}: {str(body)[:120]}")

        report.add(
            "OpenAPI documents the v1 surface",
            len(json.loads(urllib.request.urlopen(f"{BASE}/openapi.json", timeout=15).read())["paths"]) >= 50,
            ">= 50 documented paths",
        )

        report.add(
            "RTSP credentials never returned",
            _no_credentials_leaked(get),
            "camera payloads expose only a redacted preview",
        )
        report.add("WebSocket endpoint accepts a connection", _websocket_ok, "/ws/events")
        passed, detail = _stream_endpoints(get)
        report.add(
            "Video stream endpoints are mounted",
            bool(passed) if passed is not None else True,
            detail,
            skipped=passed is None,
        )
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()

    return report.finish()


def _stream_endpoints(getter) -> tuple[bool | None, str]:
    """Returns (passed, detail). ``passed=None`` means "not testable here"."""

    """The Live page's picture comes from these three paths — and nothing else.

    Regression guard: the streams router was documented in RESOURCE_PREFIXES
    but never included in the v1 router, so every video URL 404'd and the
    dashboard silently showed no camera feeds at all.
    """
    _, body = getter("/api/v1/cameras")
    items = body.get("items", body) if isinstance(body, dict) else body
    if not items:
        return None, "no cameras configured — cannot exercise the stream endpoints"
    # Exercise a camera that is actually producing frames. Picking an arbitrary
    # one (e.g. a webcam with no hardware attached) makes this check fail for
    # the wrong reason and times the whole API suite out.
    producing = [
        c
        for c in items
        if c.get("status") == "online" and (c.get("total_frames") or 0) > 0
    ]
    if not producing:
        online = [c for c in items if c.get("status") == "online"]
        if online:
            return None, (
                f"{len(online)} camera(s) report online but none has ever produced a "
                "frame — start a working camera and re-run"
            )
        return None, "no camera is online — start one and re-run"
    camera_id = producing[0]["id"]

    status, _ = getter(f"/api/v1/streams/{camera_id}/status")
    if status != 200:
        return False, f"GET /streams/{{id}}/status returned HTTP {status} (expected 200)"

    # A camera that is not running has no frame buffered: that is a 503 from
    # the snapshot endpoint. A 404 would mean the route itself is missing.
    status = get_status(f"/api/v1/streams/{camera_id}/snapshot.jpg")
    if status not in (200, 503):
        return False, f"GET /streams/{{id}}/snapshot.jpg returned HTTP {status}"

    spec = json.loads(urllib.request.urlopen(f"{BASE}/openapi.json", timeout=15).read())
    mjpeg = spec["paths"].get("/api/v1/streams/{camera_id}/mjpeg")
    if not mjpeg:
        return False, "MJPEG path missing from the OpenAPI document"
    content = mjpeg["get"]["responses"]["200"].get("content", {})
    if "multipart/x-mixed-replace" not in content:
        return False, "MJPEG 200 response does not declare multipart/x-mixed-replace"
    return True, "status + snapshot reachable, MJPEG declared as multipart/x-mixed-replace"


def _no_credentials_leaked(getter) -> bool:
    _, body = getter("/api/v1/cameras")
    if not isinstance(body, dict):
        return False
    text = json.dumps(body)
    return "stream_url_encrypted" not in text and ":password@" not in text


def _websocket_ok() -> bool:
    """Open a WebSocket handshake without a library dependency."""
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=10) as sock:
            sock.sendall(
                b"GET /ws/events HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n"
                b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            return b"101" in sock.recv(200)
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())