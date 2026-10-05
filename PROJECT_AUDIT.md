# IRIS — PROJECT AUDIT

**Audit date:** 2026-10-03
**Audited by:** Lead architect (automated engineering agent)
**Scope:** Full workspace inspection + host environment inspection
**Status legend:** `OK` present and verified · `MISSING` not present · `PARTIAL` present but insufficient

---

## 1. Current project state

| Item | Finding |
|---|---|
| Workspace root | `C:\iris` (Git Bash: `/c/iris`) |
| Files before audit | **0 files, 0 directories** — completely empty |
| Existing implementation | **None.** This is a greenfield build. No existing code was deleted. |
| Existing README | Supplied externally at `C:\Users\LENOVO\Downloads\README.md` (specification source) |
| Git repository | Not initialised at audit time |
| Existing tests | None |
| Existing DB / models / datasets | None |
| Existing Docker config | None |

**Conclusion:** No existing working implementation exists, therefore nothing was at risk of being overwritten. All code in this repository was authored during this engagement.

---

## 2. Detected environment

### 2.1 Operating system

| Property | Value |
|---|---|
| OS | Microsoft Windows 11 Pro |
| Build | 10.0.26300 (64-bit, Multiprocessor Free) |
| Manufacturer | Dell Inc. (laptop) |
| Shell used for tooling | Git Bash (`/usr/bin/bash`) on Windows |

> Note: `wmic` is absent on this Windows build. Hardware was detected with `systeminfo`,
> `platform` and `psutil` instead. All project code must not depend on `wmic`.

### 2.2 CPU / memory / storage

| Property | Value | Notes |
|---|---|---|
| CPU | Intel 64 Family 6 Model 140 Stepping 1 (Alder Lake class) | No AVX-512-specific path assumed |
| Torch threads | 4 (`torch.get_num_threads()`) | 4 physical/logical cores visible to PyTorch |
| Total RAM | 16,106 MB (~15.7 GiB) | |
| Available RAM at audit | 9,684 MB (~9.5 GiB) | Constrains how many concurrent model instances are viable |
| Disk total | 254.6 GB | |
| Disk free | 114.1 GB | Sufficient for weights, clips, SQLite DB |

### 2.3 GPU / acceleration

| Provider | Status |
|---|---|
| CUDA | **Not available.** `torch.cuda.is_available()` → `False`; `torch 2.14.0+cpu` build |
| NVIDIA driver | **Not present.** `nvidia-smi` not found, no NVIDIA Video Controller |
| Apple MPS | Not applicable (Windows) |
| ONNX Runtime | **OK**, providers = `['AzureExecutionProvider', 'CPUExecutionProvider']` |
| OpenVINO / TensorRT | Not installed |

**Design consequence:** IRIS must run **CPU-first**. GPU paths are implemented and auto-detected
but cannot be validated on this host. All benchmark numbers recorded in this project are CPU numbers
and are labelled as such.

### 2.4 Language runtimes

| Runtime | Version | Action |
|---|---|---|
| Python | 3.14.7 | Usable; venv created with `--system-site-packages` to reuse the preinstalled AI stack |
| Node.js | v24.20.0 | Usable for frontend |
| npm | 11.19.0 | Package manager = npm |
| Git | 2.55.0.windows.5 | Initialised in this repo |

---

## 3. Detected dependencies (preinstalled on host)

Verified by import test and `pip list`:

| Package | Version | Status | Use in IRIS |
|---|---|---|---|
| fastapi | 0.141.1 | OK | Backend API |
| uvicorn | 0.52.4 | OK | ASGI server |
| pydantic | 2.13.5 | OK | Schemas / settings |
| pydantic-settings | 2.15.0 | OK | `.env` configuration |
| SQLAlchemy | 2.0.52 | OK | ORM |
| alembic | 1.19.2 | OK | Migrations |
| torch | 2.14.0+cpu | OK | Inference runtime |
| torchvision | 0.29.0 | OK | Vision ops |
| ultralytics | 8.4.166 | OK | YOLO11 detector + ByteTrack/BoT-SORT tracker |
| lap | 0.5.13 | OK | Assignment matrix required by ByteTrack |
| opencv-python(-contrib/-headless) | 5.0.0.93 | OK | Decode, preprocessing, geometry |
| numpy | 2.5.2 | OK | Arrays |
| onnxruntime | 1.29.0 | OK | Optional inference provider |
| scipy / scikit-learn | 1.18.1 / 1.9.0 | OK | Metrics, statistical eval |
| pandas | 3.0.5 | OK | Reporting / eval tables |
| Pillow | 12.3.0 | OK | Image IO |
| psutil | 7.2.2 | OK | System health metrics |
| PyYAML | 6.0.3 | OK | Config / model registry files |
| duckdb | 1.5.5 | OK | Optional analytical queries |
| httpx | 0.28.1 | OK | Test client, async HTTP |
| pytest / pytest-asyncio | 9.1.1 / 1.4.0 | OK | Test suite |
| websockets | 17.1 | OK | WebSocket server support |
| python-multipart | present | OK | File uploads |

### 3.1 Dependencies that were missing and installed into `backend/.venv`

| Package | Version | Reason |
|---|---|---|
| PyJWT | 2.15.1 | JWT access tokens (spec §10.1 requires real password hashing + session tokens) |
| bcrypt | 5.0.0 | Modern password hashing (spec §10.1) |

The venv was created with `--system-site-packages` so torch/ultralytics/opencv are reused
rather than re-downloaded (~2 GB). Only the two missing packages were installed.

### 3.2 Network access

| Check | Result |
|---|---|
| HTTPS to github.com | **OK** (HTTP 200) |
| PyPI install | **OK** (`PyJWT` resolved and downloaded) |
| `curl` binary download integrity | **UNRELIABLE** — a `curl -L` download of `yolo11n.pt` produced a 2,666,496-byte truncated file that PyTorch rejected as corrupt. The same URL via Python `urllib` streaming produced the correct 5,613,764-byte file. **All model downloads in IRIS therefore use Python streaming, never `curl`.** |

---

## 4. Existing implementation

**None.** Confirmed by directory listing (empty) at session start.

---

## 5. Missing components (all of them)

Every component below was missing and had to be authored:

1. Repository structure and `.gitignore`
2. Configuration layer (`.env`, `pydantic-settings`)
3. Structured logging (app / AI / camera / security channels)
4. FastAPI application, routers, OpenAPI, error handling
5. Database engine, session, ORM models, Alembic migrations
6. Auth (bcrypt + JWT), RBAC (admin/manager/staff/service), audit log
7. Camera abstraction: webcam, RTSP, video file, image
8. Capture lifecycle, frame sampling, bounded queues, reconnect with backoff
9. Camera health (offline / frozen frame / dropped frames)
10. Authorised MJPEG stream proxy
11. Model registry and model interface
12. Product / person / shelf detector wrapper (YOLO11)
13. SKU recognition via embedding retrieval against the product catalog
14. ByteTrack multi-object tracking wrapper
15. Shelf-zone mapping, occupancy, void detection
16. Stock estimation engine with confidence
17. Rules engine + alert lifecycle with dedup/cooldown
18. Planogram compliance engine
19. Footfall / line crossing / dwell / occupancy analytics
20. Queue analytics with configurable thresholds
21. Event service + WebSocket gateway
22. Analytics aggregation and CSV export
23. Video upload → processing job pipeline
24. System health / observability
25. Retention & cleanup service
26. React + TypeScript + Vite + Tailwind frontend (10 pages)
27. Seed data
28. Test suites (unit / API / AI / integration / frontend)
29. Verification scripts (`verify_*.py`, `run_all_tests.py`)
30. Docker deployment files
31. Documentation set

---

## 6. Implementation risks and mitigations

| # | Risk | Severity | Mitigation actually applied |
|---|---|---|---|
| R1 | **No trained retail SKU model.** No SKU-110K/Products-10K weights locally; fine-tuning needs hours of CPU and store-specific data. | High | Ship a two-stage real pipeline: COCO-pretrained YOLO11n provides **real** `bottle/shelf/cup/book` etc. person detection; SKU resolution is done by **real DINOv2-class embedding retrieval** against a catalog of product reference images, with an explicit `unknown` state. Registered as `BASELINE` status, never claimed as validated retail accuracy. Training scripts provided. |
| R2 | **No store footage.** Cannot validate against a real shelf. | High | Provide a synthetic-but-real-media test fixture generator (a physically plausible shelf scene rendered with OpenCV) used **only** for pipeline/regression tests, clearly labelled DEMO. All accuracy claims are withheld. |
| R3 | **No physical RTSP camera / webcam.** | Medium | Full RTSP + webcam code paths implemented. Validation performed against (a) video-file sources, (b) an **invalid** RTSP URL to exercise failure/reconnect logic, (c) an in-process MJPEG loopback stream. Hardware validation recorded as BLOCKED. |
| R4 | **CPU-only host.** Inference ~8 FPS at 640px for YOLO11n. | Medium | Bounded queues with stale-frame dropping; configurable per-task sampling rates; small model default (yolo11n); latency/FPS measured and published; GPU paths written but unverifiable. |
| R5 | **Python 3.14 is very new.** Some wheels may be unavailable. | Medium | Verified every needed package imports and runs inference. Used `--system-site-packages` venv. Documented 3.14 as the validated interpreter. |
| R6 | **16 GB RAM / 4 threads** limits multi-camera concurrency. | Medium | One shared detector instance per task type with a bounded work queue; per-camera concurrency capped; no per-camera model duplication. |
| R7 | **Frontend build time / disk.** npm install of Tailwind+Recharts. | Low | Pinned versions, verified `npm run build` (tsc + vite). |
| R8 | **Curl download corruption** (see §3.2). | Medium | All downloads implemented in Python with checksum reporting. |
| R9 | **Accidental secret leakage** (RTSP URLs in API/frontend/logs). | High | Credentials stored encrypted-at-rest field, redacted in every API response and log formatter, separate `/api/v1/cameras/{id}/stream` proxy endpoint. Automated test asserts the secret never appears in API output or logs. |
| R10 | **False success claims.** | High | `docs/FINAL_ACCEPTANCE_MATRIX.md` requires an evidence column; automated verification scripts print measured results only; every status distinguishes Implemented / Tested / Verified / Hardware-validated. |

---

## 7. Available hardware acceleration summary

```
CUDA GPU .............. NOT AVAILABLE (torch CPU build, no NVIDIA driver)
MPS .................. NOT APPLICABLE (Windows)
ONNX Runtime ......... AVAILABLE (CPU execution provider only)
OpenVINO ............. not installed
TensorRT ............. not installed
OpenCV video backend . available (FFmpeg-backed; GStreamer presence recorded at runtime)
```

**IRIS runs on CPU. Any statement about GPU performance in this project is marked UNVERIFIED.**

---

## 8. Recommended execution order

Derived from README §13/§17 and adapted to the detected environment:

| Order | Phase | Gate before proceeding |
|---|---|---|
| 0 | Audit | This document + `docs/IMPLEMENTATION_STATUS.md` exist |
| 1 | Foundation: config, logging, DB, migrations, FastAPI, auth | `GET /api/system/health` returns `healthy` from a live server |
| 2 | Camera pipeline: sources, sampling, health, reconnect, proxy | A real video file is decoded; invalid source produces a visible failure + recovery |
| 3 | AI core: registry, detector, tracker, embeddings, inference service | Measured FPS/latency recorded on a real media file |
| 4 | Retail intelligence: catalog, shelves, stock estimation, rules, alerts | Controlled fixture produces an estimate and a threshold alert |
| 5 | Customer + queue analytics | Line-crossing/dwell/queue metrics computed from a real tracked sequence |
| 6 | Frontend shell + pages + WebSocket | `npm run build` passes; every page renders real or explicitly-empty data |
| 7 | Integration | End-to-end test: source → inference → rules → event → DB → WS → API |
| 8 | Hardening: security, privacy, retention, backup, upload validation | Security test suite passes |
| 9 | QA, docs, acceptance matrix, final report | `run_all_tests.py` green and evidence captured |

---

## 9. Audit conclusion

The workspace was empty; there was no existing implementation to preserve. The host has a
**complete, working CPU AI stack** (torch, ultralytics, OpenCV 5, ByteTrack's `lap`) and a
current Node toolchain, plus working network access. That is sufficient to build and **actually
verify** the complete IRIS software stack on CPU. It is **not** sufficient to validate retail
SKU recognition accuracy, RTSP hardware behaviour, or GPU acceleration — those remain explicitly
BLOCKED / BASELINE and are documented as such throughout `docs/`.