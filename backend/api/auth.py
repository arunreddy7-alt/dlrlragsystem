"""Authentication routes and user dependency."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from backend.core.database import db
from backend.core.security import create_access_token, get_token_subject, hash_password, verify_password
from backend.models.schemas import TokenResponse, UserCreate, UserLogin, UserResponse

router = APIRouter(tags=["auth"])


def row_to_user(row: sqlite3.Row) -> UserResponse:
    """Convert a SQLite row into a user response."""

    return UserResponse(id=row["id"], username=row["username"], email=row["email"], created_at=row["created_at"])


def get_current_user(subject: str = Depends(get_token_subject)) -> sqlite3.Row:
    """Return the authenticated user."""

    try:
        user_id = int(subject)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject") from exc
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate) -> TokenResponse:
    """Register a local user and return a JWT."""

    try:
        password_hash = hash_password(payload.password)
        user = db.create_user(payload.username, payload.email, password_hash)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username or email already exists") from exc
    return TokenResponse(access_token=create_access_token(str(user["id"])))


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin) -> TokenResponse:
    """Authenticate a user and return a JWT."""

    user = db.get_user_by_username(payload.username)
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    return TokenResponse(access_token=create_access_token(str(user["id"])))


@router.get("/me", response_model=UserResponse)
def me(user: sqlite3.Row = Depends(get_current_user)) -> UserResponse:
    """Return authenticated user profile."""

    return row_to_user(user)
