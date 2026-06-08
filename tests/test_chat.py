"""Chat citation parsing tests."""

from backend.api.chat import _sources_from_answer


def test_sources_from_answer() -> None:
    answer = "Answer text\n\nSources:\n* guide.pdf (Page 15)\n* guide.pdf (Page 15)\n* handbook.pdf (Page 2)"
    sources = _sources_from_answer(answer)
    assert len(sources) == 2
    assert sources[0].filename == "guide.pdf"
    assert sources[0].page_number == 15
    assert sources[1].filename == "handbook.pdf"
