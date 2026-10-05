"""Application error types and FastAPI exception handlers.

Every failure surfaces to clients as a consistent JSON envelope::

    {"error": {"code": "not_found", "message": "...", "request_id": "..."}}

Internal exception types and stack traces are logged but **never** returned to
the client in production.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_var

logger = get_logger("app")

# Starlette renamed this constant; support both without emitting a warning.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


class IRISError(Exception):
    """Base class for all IRIS business errors."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "bad_request"
    message: str = "Request could not be processed."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(IRISError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "The requested resource does not exist."


class ConflictError(IRISError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "The request conflicts with the current state."


class ValidationError(IRISError):
    status_code = HTTP_422
    code = "validation_error"
    message = "The request payload failed validation."


class AuthError(IRISError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "Authentication is required."


class PermissionError_(IRISError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"
    message = "You do not have permission to perform this action."


class RateLimitError(IRISError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Too many requests. Please slow down."


class SourceUnavailableError(IRISError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "source_unavailable"
    message = "The video source is not currently readable."


class ModelError(IRISError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_error"
    message = "The requested model is unavailable."


class UploadValidationError(ValidationError):
    code = "upload_rejected"
    message = "The uploaded file was rejected."


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id_var.get(),
        }
    }
    if details:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every IRIS error handler to the application."""

    @app.exception_handler(IRISError)
    async def _iris_error(_: Request, exc: IRISError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=HTTP_422,
            content=_envelope(
                "validation_error",
                "The request payload failed validation.",
                {"errors": _sanitise_errors(exc.errors())},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 401: "unauthorized", 403: "forbidden", 429: "rate_limited"}
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(codes.get(exc.status_code, "http_error"), str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception",
            extra={"path": request.url.path, "method": request.method},
        )
        from app.core.config import settings

        message = (
            str(exc)
            if settings.debug or settings.env == "development"
            else "An internal error occurred. Check the server logs for details."
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", message),
        )


def _sanitise_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop non-serialisable context values from pydantic error payloads."""
    cleaned: list[dict[str, Any]] = []
    for err in errors[:25]:
        cleaned.append(
            {
                "loc": [str(part) for part in err.get("loc", [])],
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
        )
    return cleaned