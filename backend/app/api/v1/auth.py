"""Authentication endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from sqlalchemy import func

from app.api.deps import CurrentUser, DbSession, write_audit
from app.core.config import settings
from app.core.errors import AuthError, NotFoundError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    hash_password,
    needs_rehash,
    token_expiry_seconds,
    verify_password,
)
from app.models import User
from app.schemas import LoginRequest, MessageResponse, TokenResponse, UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger("app")
security_log = get_logger("security")


@router.post("/login", response_model=TokenResponse, summary="Authenticate and obtain an access token")
def login(payload: LoginRequest, request: Request, session: DbSession) -> TokenResponse:
    user = session.query(User).filter(func.lower(User.username) == payload.username.lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        security_log.warning(
            "Failed login attempt",
            extra={"username": payload.username, "ip": request.client.host if request.client else None},
        )
        raise AuthError("Invalid username or password.")
    if not user.is_active:
        raise AuthError("This account is disabled.")

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    user.last_login_at = datetime.now(tz=timezone.utc)
    session.flush()
    token = create_access_token(user.id, role=user.role, extra={"username": user.username})
    security_log.info("Login succeeded", extra={"username": user.username})
    return TokenResponse(
        access_token=token,
        expires_in=token_expiry_seconds(),
        user=UserRead.model_validate(user),
    )


@router.get("/me", response_model=UserRead, summary="Current authenticated user")
def me(principal: CurrentUser, session: DbSession) -> UserRead:
    if principal.user is None:
        # Auth disabled (development): return the configured local operator.
        return UserRead(
            id="local-dev",
            username="local-dev",
            full_name="Local developer (auth disabled)",
            email=None,
            role=principal.role,
            is_active=True,
            last_login_at=None,
            created_at=datetime.now(tz=timezone.utc),
        )
    return UserRead.model_validate(principal.user)


@router.get("/config", summary="Authentication mode and requirements")
def auth_config(principal: CurrentUser) -> dict:
    return {
        "auth_enabled": settings.auth_enabled,
        "token_algorithm": settings.jwt_algorithm,
        "expires_in_seconds": token_expiry_seconds(),
        "password_min_length": settings.password_min_length,
        "roles": ["admin", "manager", "staff", "service"],
        "current_role": principal.role,
        "note": (
            "Authentication is disabled for local development. Set IRIS_AUTH_ENABLED=true "
            "before exposing IRIS beyond the local machine."
        ),
    }


@router.post("/password", response_model=MessageResponse, summary="Change the current user's password")
def change_password(
    current_password: str,
    new_password: str,
    principal: CurrentUser,
    request: Request,
    session: DbSession,
) -> MessageResponse:
    if principal.user is None:
        raise AuthError("Password changes require authentication to be enabled.")
    if not verify_password(current_password, principal.user.password_hash):
        security_log.warning("Password change rejected", extra={"username": principal.username})
        raise AuthError("The current password is incorrect.")
    principal.user.password_hash = hash_password(new_password)
    write_audit(session, principal, "user.password_changed", entity_type="user",
                entity_id=principal.user.id, request=request)
    return MessageResponse(message="Password updated.")


@router.post("/logout", response_model=MessageResponse, summary="Log out (token is stateless)")
def logout(principal: CurrentUser) -> MessageResponse:
    security_log.info("Logout", extra={"username": principal.username})
    return MessageResponse(
        message="Logged out. JWT access tokens are stateless: discard the token client-side."
    )


@router.get("/users", response_model=list[UserRead], summary="List user accounts (admin)")
def list_users(session: DbSession, _: CurrentUser) -> list[UserRead]:
    return [UserRead.model_validate(u) for u in session.query(User).order_by(User.username).all()]


@router.post("/users", response_model=UserRead, status_code=201, summary="Create a user (admin)")
def create_user(
    payload: UserCreate, request: Request, session: DbSession, principal: CurrentUser
) -> UserRead:
    from app.core.security import Role

    existing = session.query(User).filter(User.username == payload.username).first()
    if existing is not None:
        from app.core.errors import ConflictError

        raise ConflictError(f"Username {payload.username!r} already exists.")
    if payload.role == Role.ADMIN.value and principal.role != Role.ADMIN.value:
        from app.core.errors import PermissionError_

        raise PermissionError_("Only an admin can create another admin.")
    user = User(
        username=payload.username,
        full_name=payload.full_name,
        email=payload.email,
        role=payload.role,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    session.flush()
    write_audit(session, principal, "user.created", entity_type="user", entity_id=user.id,
                detail={"username": user.username, "role": user.role}, request=request)
    return UserRead.model_validate(user)


@router.get("/users/{user_id}", response_model=UserRead, summary="Fetch a user (admin)")
def get_user(user_id: str, session: DbSession, _: CurrentUser) -> UserRead:
    user = session.get(User, user_id)
    if user is None:
        raise NotFoundError(f"User {user_id!r} does not exist.")
    return UserRead.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserRead, summary="Update a user (admin)")
def update_user(
    user_id: str,
    payload: UserUpdate,
    request: Request,
    session: DbSession,
    principal: CurrentUser,
) -> UserRead:
    user = session.get(User, user_id)
    if user is None:
        raise NotFoundError(f"User {user_id!r} does not exist.")
    data = payload.model_dump(exclude_unset=True)
    password = data.pop("password", None)
    for key, value in data.items():
        if value is not None:
            setattr(user, key, value)
    if password:
        user.password_hash = hash_password(password)
    session.flush()
    write_audit(session, principal, "user.updated", entity_type="user", entity_id=user.id,
                detail={"fields": sorted(data.keys())}, request=request)
    return UserRead.model_validate(user)