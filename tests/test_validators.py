import pytest

from core.validators import (
    MAX_TEXT_LENGTH,
    MAX_URL_LENGTH,
    validate_indian_mobile,
    validate_port,
    validate_text,
    validate_upi_address,
    validate_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "HTTPS://Example.test/Agenda?Ticket=A%2FB#Section",
        "http://example.test.:8080/",
        "https://sahayata.भारत/सहायता",
        "http://127.0.0.1:8080/Check",
        "https://[2001:db8::1]:443/Agenda",
        "https://example.test/" + "a" * (MAX_URL_LENGTH - len("https://example.test/")),
    ],
)
def test_valid_url_is_unchanged(url: str) -> None:
    assert validate_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "/Agenda",
        "ftp://example.test/",
        "javascript:alert(1)",
        "https:///Agenda",
        "https://user:secret@example.test/",
        "https://example.test:/",
        "https://example.test:0/",
        "https://example.test:65536/",
        "https://example.test:४४३/",
        "https://example.test\\@other.test/",
        "https://bad..example/",
        "https://bad_label.test/",
        "https://example.test/a b",
        "https://example.test/?value=<tag>",
        'https://example.test/?value="',
        "https://exa\nmple.test/",
        "https://example.test/\u200f",
        "https://example.test/%XYZ",
        "https://[2001:db8::1]spoof.test",
        "https://[fe80::1%25eth0]/",
        "https://2001:db8::1/",
        "http://127.1/",
        "http://0x7f000001/",
        "http://256.0.0.1/",
        "https://example.test/" + "a" * MAX_URL_LENGTH,
        b"https://example.test/",
    ],
)
def test_invalid_url_is_rejected(url: object) -> None:
    with pytest.raises(ValueError):
        validate_url(url)


@pytest.mark.parametrize(("port", "expected"), [(1, 1), ("443", 443), (65_535, 65_535)])
def test_valid_port(port: int | str, expected: int) -> None:
    assert validate_port(port) == expected


@pytest.mark.parametrize("port", [False, True, 0, -1, 65_536, 443.0, "", " 443", "+443", "４４３"])
def test_invalid_port(port: object) -> None:
    with pytest.raises(ValueError, match="Port"):
        validate_port(port)


@pytest.mark.parametrize(
    "number", ["9876543210", "+91 98765-43210", "00919876543210", "919876543210", "09876543210"]
)
def test_indian_mobile_is_normalized(number: str) -> None:
    assert validate_indian_mobile(number) == "+919876543210"


@pytest.mark.parametrize(
    "number",
    ["", "1930", "5876543210", "+1 9876543210", "98765432101", "98765\n43210", "९८७६५४३२१०"],
)
def test_invalid_phone(number: str) -> None:
    with pytest.raises(ValueError, match="Indian mobile"):
        validate_indian_mobile(number)


def test_text_preserves_multilingual_unicode_and_whitespace() -> None:
    message = "  कृपया ध्यान दें\nமோசடி\tQR  "
    assert validate_text(message) == message
    assert validate_text("x" * MAX_TEXT_LENGTH) == "x" * MAX_TEXT_LENGTH


@pytest.mark.parametrize("value", ["", " \t ", "bad\x00text", "\ud800", "x" * (MAX_TEXT_LENGTH + 1)])
def test_invalid_text(value: str) -> None:
    with pytest.raises(ValueError):
        validate_text(value)


@pytest.mark.parametrize("value", ["Recipient.01-3@Bank", "9990001112@upi", "a_b@bank.provider"])
def test_upi_address_preserves_case(value: str) -> None:
    assert validate_upi_address(value) == value


@pytest.mark.parametrize(
    "value", ["a", "@bank", "a@@bank", "a@bank..provider", "a@bank-", "a b@bank", "a@bank\n"]
)
def test_invalid_upi_address_is_not_in_error(value: str) -> None:
    with pytest.raises(ValueError, match="UPI virtual") as error:
        validate_upi_address(value)
    assert repr(value) not in str(error.value)
