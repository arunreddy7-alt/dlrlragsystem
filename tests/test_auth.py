"""Authentication helper tests."""

from backend.core.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_and_verify() -> None:
    password_hash = hash_password("local-password")
    assert password_hash != "local-password"
    assert verify_password("local-password", password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_jwt_round_trip() -> None:
    token = create_access_token("42")
    assert decode_access_token(token) == "42"
