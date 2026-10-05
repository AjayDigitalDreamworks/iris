"""FastAPI dependencies: authentication, RBAC, pagination, audit and store scope."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Callable

from fastapi import Depends, Header, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AuthError, PermissionError_, RateLimitError
from app.core.logging import get_logger, new_request_id, request_id_var
from app.core.security import (
    SlidingWindowLimiter,
    decode_access_token,
    role_at_least,
)
from app.db.session import get_db
from app.models import AuditLog, Store, User

logger = get_logger("app")
security_log = get_logger("security")

bearer_scheme = HTTPBearer(auto_error=False)

#: Per-process limiter applied to sensitive endpoints.
rate_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)


def flush_or_conflict(session: Any, subject: str) -> None:
    """Flush, translating a uniqueness violation into a clean 409.

    Without this, submitting a duplicate raises a raw ``IntegrityError`` that
    escapes as a 500 with a database-level message.  Operators get an
    actionable conflict instead, and no SQL detail leaks to the client.
    """
    from sqlalchemy.exc import IntegrityError

    from app.core.errors import ConflictError

    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise ConflictError(
            f"A {subject} with these unique details already exists."
        ) from exc


def commit_before_camera_work(session: Any) -> None:
    """Commit the request transaction before doing camera lifecycle work.

    Starting, stopping or restarting a camera pipeline makes a worker thread
    begin writing frames to the database immediately. If the request session is
    still holding its write transaction, that worker blocks on the single
    SQLite writer lock until the busy timeout expires, and the API call stalls
    for minutes before the request's own dependency teardown can commit.
    Committing first breaks that cycle.

    ``get_db`` commits again during teardown, which is a harmless no-op once the
    transaction is already committed.
    """
    try:
        session.commit()
    except Exception:  # pragma: no cover - let the handler surface real errors
        session.rollback()
        raise


@dataclass
class Principal:
    """The authenticated caller."""

    user: User | None
    role: str
    username: str

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None

    @property
    def is_service(self) -> bool:
        return self.role == "service"


ANONYMOUS = Principal(user=None, role="anonymous", username="anonymous")


def get_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> Principal:
    """Resolve the caller from a bearer token, honouring the auth switch.

    When ``IRIS_AUTH_ENABLED`` is false (development default) the API is open
    and the principal is an admin-equivalent local operator.  In production
    auth must be enabled; :func:`app.core.config.Settings.warn_if_insecure`
    flags it at startup.
    """
    request_id_var.set(request.headers.get("x-request-id") or new_request_id())
    if not settings.auth_enabled:
        return Principal(user=None, role="admin", username="local-dev")
    if credentials is None or not credentials.credentials:
        raise AuthError("A bearer token is required.")
    try:
        payload = decode_access_token(credentials.credentials)
    except AuthError as exc:
        security_log.warning(
            "Rejected request with invalid token",
            extra={"path": request.url.path, "method": request.method},
        )
        raise exc
    subject = payload.get("sub", "")
    role = payload.get("role", "staff")

    from app.services.db_service import run_in_session

    user = run_in_session(lambda session: session.get(User, subject))
    if user is None or not user.is_active:
        raise AuthError("The account for this token no longer exists or is disabled.")
    return Principal(user=user, role=user.role, username=user.username)


def require_role(minimum: str) -> Callable[..., Principal]:
    """Dependency factory enforcing a minimum role."""

    def _dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if not role_at_least(principal.role, minimum):
            security_log.warning(
                "Authorization denied",
                extra={"role": principal.role, "required": minimum, "user": principal.username},
            )
            raise PermissionError_(f"This action requires the '{minimum}' role.")
        return principal

    return _dependency


require_admin = require_role("admin")
require_manager = require_role("manager")
require_staff = require_role("staff")


def rate_limit(key: str | None = None) -> Callable[..., None]:
    """Dependency factory applying a sliding-window rate limit."""

    def _dependency(request: Request) -> None:
        client = key or (request.client.host if request.client else "unknown")
        if not rate_limiter.allow(f"{client}:{request.url.path}", datetime.now(tz=timezone.utc).timestamp()):
            security_log.warning("Rate limit exceeded", extra={"client": client, "path": request.url.path})
            raise RateLimitError()

    return _dependency


class Pagination:
    """Bounded pagination parameters."""

    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> None:
        self.limit = limit
        self.offset = offset


def get_pagination(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Pagination:
    return Pagination(limit=limit, offset=offset)


DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[Principal, Depends(get_principal)]
AdminUser = Annotated[Principal, Depends(require_admin)]
ManagerUser = Annotated[Principal, Depends(require_manager)]
StaffUser = Annotated[Principal, Depends(require_staff)]
PageParams = Annotated[Pagination, Depends(get_pagination)]


def get_default_store(session: DbSession) -> Store | None:
    """Resolve the store used when a request does not name one."""
    store_id = request_store_id.get()
    if store_id:
        store = session.get(Store, store_id)
        if store is not None:
            return store
    return session.query(Store).order_by(Store.created_at.asc()).first()


request_store_id: Any = None


def write_audit(
    session: Session,
    principal: Principal,
    action: str,
    *,
    entity_type: str | None = None,
    entity_id: str | None = None,
    detail: dict[str, Any] | None = None,
    request: Request | None = None,
) -> None:
    """Append an audit row. Never stores secrets (values are redacted upstream)."""
    try:
        session.add(
            AuditLog(
                user_id=principal.user.id if principal.user else None,
                username=principal.username,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                detail=detail or {},
                ip_address=request.client.host if request and request.client else None,
                request_id=request_id_var.get(),
            )
        )
        session.flush()
        security_log.info(
            "Audit action recorded",
            extra={"action": action, "entity_type": entity_type, "entity_id": entity_id,
                   "user": principal.username},
        )
    except Exception as exc:  # pragma: no cover - auditing must not break the action
        logger.warning("Audit write failed", exc_info=exc)


def current_user_header(x_request_id: Annotated[str | None, Header()] = None) -> None:
    """Convenience dependency placeholder for request-id propagation."""
    if x_request_id:
        request_id_var.set(x_request_id)