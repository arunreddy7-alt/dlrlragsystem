"""Upload validation tests."""

import pytest
from fastapi import HTTPException

from backend.api.upload import _safe_filename


def test_safe_filename_accepts_pdf() -> None:
    assert _safe_filename("../report.pdf") == "report.pdf"


def test_safe_filename_rejects_non_pdf() -> None:
    with pytest.raises(HTTPException):
        _safe_filename("report.txt")
