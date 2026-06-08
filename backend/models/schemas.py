"""Pydantic v2 request and response models."""

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(min_length=5, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=72)


class UserLogin(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: str


class DocumentResponse(BaseModel):
    id: int
    user_id: int
    filename: str
    upload_time: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    filename: str
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class ChatHistoryItem(BaseModel):
    id: int
    user_id: int
    role: str
    message: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    app: str
