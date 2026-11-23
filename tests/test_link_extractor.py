import pytest

from core.extractors.link_extractor import extract_links
from core.validators import MAX_TEXT_LENGTH


def test_preserves_case_encoding_and_first_seen_order() -> None:
    text = (
        "Read HTTPS://Example.test/Agenda?Id=ABC%2fXYZ#Top, then https://example.test/agenda. "
        "Again HTTPS://Example.test/Agenda?Id=ABC%2fXYZ#Top."
    )
    assert extract_links(text) == (
        "HTTPS://Example.test/Agenda?Id=ABC%2fXYZ#Top",
        "https://example.test/agenda",
    )


def test_trims_prose_without_removing_balanced_url_delimiters() -> None:
    text = "[See (https://example.test/a(b))]. Also “https://[2001:db8::1]”, or https://example.test/Next!"
    assert extract_links(text) == (
        "https://example.test/a(b)",
        "https://[2001:db8::1]",
        "https://example.test/Next",
    )


def test_skips_malformed_and_non_http_candidates() -> None:
    text = (
        "https://user@bad.test https://bad.test:abc ftp://example.test "
        "unauthorizedhttps://example.test http://good.test/Path"
    )
    assert extract_links(text) == ("http://good.test/Path",)


def test_explicit_delimiters_preserve_url_punctuation() -> None:
    assert extract_links(
        "<https://example.test/Agenda?Token=A!> "
        "\"http://example.test/O'Neil!?\" ‘https://example.test/Latest!’"
    ) == (
        "https://example.test/Agenda?Token=A!",
        "http://example.test/O'Neil!?",
        "https://example.test/Latest!",
    )


@pytest.mark.parametrize("value", ["", " \n ", "A message with no link", "upi://pay?pa=payee@bank"])
def test_link_free_text(value: str) -> None:
    assert extract_links(value) == ()


@pytest.mark.parametrize("fill", ["a", " "])
def test_extractor_checks_input_length(fill: str) -> None:
    with pytest.raises(ValueError):
        extract_links(fill * (MAX_TEXT_LENGTH + 1))
