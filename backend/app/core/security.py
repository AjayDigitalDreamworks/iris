"""Authentication, password hashing, RBAC and credential protection.

* Passwords are hashed with **bcrypt** (never stored or logged in clear text).
* Access tokens are signed **JWT (HS256)** with an expiry.
* Authorisation is role based: ``admin > manager > staff`` plus a least-privilege
  ``service`` account.
* Camera RTSP credentials are encrypted at rest with Fernet and are **never**
  returned by the API — the frontend only ever receives a redacted preview.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import bcrypt
import jwt
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings
from app.core.errors import AuthError, ValidationError
from app.core.logging import get_logger

security_log = get_logger("security")


class Role(str, Enum):
    """User roles, ordered from most to least privileged."""

    ADMIN = "admin"
    MANAGER = "manager"
    STAFF = "staff"
    SERVICE = "service"


_ROLE_RANK: dict[str, int] = {r.value: i for i, r in enumerate(Role)}


def role_at_least(role: str | Role, required: str | Role) -> bool:
    """Return True when ``role`` is at least as privileged as ``required``."""
    return _ROLE_RANK.get(Role(role).value, -1) <= _ROLE_RANK.get(Role(required).value, 99)


# ── Password hashing ─────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    _validate_password_strength(password)
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verify a plaintext password against a bcrypt hash."""
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _validate_password_strength(password: str) -> None:
    if len(password) < settings.password_min_length:
        raise ValidationError(
            f"Password must be at least {settings.password_min_length} characters."
        )
    classes = sum(
        [
            any(c.islower() for c in password),
            any(c.isupper() for c in password),
            any(c.isdigit() for c in password),
            any(not c.isalnum() for c in password),
        ]
    )
    if classes < 2:
        raise ValidationError(
            "Password must combine at least two of: lowercase, uppercase, digits, symbols."
        )


def needs_rehash(password_hash: str) -> bool:
    """True when a stored hash uses weaker parameters than the current policy."""
    try:
        parts = password_hash.split("$")
        return int(parts[2]) < 12
    except (IndexError, ValueError):
        return True


# ── JWT ──────────────────────────────────────────────────────────────────────
def create_access_token(
    subject: str,
    *,
    role: str,
    extra: dict[str, Any] | None = None,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT access token for ``subject``."""
    now = datetime.now(tz=timezone.utc)
    expires = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": uuid.uuid4().hex,
        "iss": settings.app_name,
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT, raising :class:`AuthError` when invalid."""
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("The access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("The access token is invalid.") from exc


def token_expiry_seconds() -> int:
    return settings.jwt_expire_minutes * 60


# ── Request throttling ───────────────────────────────────────────────────────
class SlidingWindowLimiter:
    """Small in-process fixed-window limiter for sensitive endpoints."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        import time

        bucket = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(bucket) >= self.limit:
            self._hits[key] = bucket
            return False
        bucket.append(now)
        self._hits[key] = bucket
        if len(self._hits) > 4096:  # bounded memory
            cutoff = now - self.window
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] >= cutoff}
        return True

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


# ── Credential protection ────────────────────────────────────────────────────
def _fernet() -> Fernet:
    """Derive a Fernet key from the configured secret key (deterministic)."""
    digest = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plaintext: str | None) -> str | None:
    """Encrypt a credential for storage at rest."""
    if not plaintext:
        return None
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(ciphertext: str | None) -> str | None:
    """Decrypt a stored credential. Returns None when absent."""
    if not ciphertext:
        return None
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        security_log.error("Failed to decrypt a stored credential; it will be ignored.")
        return None


def redact_url(url: str | None) -> str | None:
    """Return a preview safe for logging and API display.

    ``rtsp://user:***@host/path``. Query-string VALUES are masked as well
    (names kept): signed CDN and media links carry their credentials in the
    query — ``?token=``, ``?auth=``, ``?X-Amz-Signature=`` — so returning the
    URL verbatim for any link without an ``@`` would leak them.
    """
    if not url:
        return None
    try:
        scheme, rest = url.split("://", 1)
    except ValueError:
        return "***"
    if "@" in rest:
        credentials, _, location = rest.rpartition("@")
        user = credentials.split(":", 1)[0]
        rest = f"{user}:***@{location}"
    return f"{scheme}://{_redact_query(rest)}"


def _redact_query(rest: str) -> str:
    """Mask query-string values, keeping parameter names visible."""
    if "?" not in rest:
        return rest
    base, _, query = rest.partition("?")
    masked = []
    for pair in query.split("&"):
        if not pair:
            continue
        key, sep, _value = pair.partition("=")
        masked.append(f"{key}{sep}***" if sep else pair)
    return f"{base}?{'&'.join(masked)}"


def split_stream_url(url: str) -> tuple[str | None, str | None, str]:
    """Split ``rtsp://user:pass@host:port/path`` into (user, password, safe_url)."""
    safe = redact_url(url) or ""
    if "://" not in url:
        return None, None, safe
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return None, None, safe
    credentials, _, _location = rest.rpartition("@")
    if ":" in credentials:
        user, _, password = credentials.partition(":")
        return user, password, safe
    return credentials, None, safe


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_api_token(nbytes: int = 32) -> str:
    """Generate a random API token (used for service accounts and share links)."""
    return secrets.token_urlsafe(nbytes)