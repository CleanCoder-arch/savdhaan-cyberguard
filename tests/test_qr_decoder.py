from decimal import Decimal
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from core.extractors import _images
from core.extractors.qr_decoder import (
    InvalidUPIPayloadError,
    MultipleQRCodesError,
    QRCodeError,
    QRCodeNotFoundError,
    decode_qr,
    parse_upi_uri,
)
from core.validators import MAX_TEXT_LENGTH


def _png(image: Image.Image) -> bytes:
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _make_qr(payload: str) -> Image.Image:
    matrix = cv2.QRCodeEncoder_create().encode(payload)
    image = Image.fromarray(np.pad(matrix, 4, constant_values=255))
    return image.resize((image.width * 7, image.height * 7), Image.Resampling.NEAREST)


def test_decode_qr_preserves_case_and_query() -> None:
    payload = "https://example.org/Agenda?ref=AbC"

    assert decode_qr(_png(_make_qr(payload))) == payload


def test_decode_qr_preserves_intentional_line_breaks() -> None:
    payload = "Line 1\nLine 2"

    assert decode_qr(_png(_make_qr(payload))) == payload


def test_decode_and_parse_real_upi_qr() -> None:
    uri = "upi://pay?pa=merchant%40upi&pn=Example+Shop&am=19.50&cu=INR&mc=0000&tr=Ref09"

    result = parse_upi_uri(decode_qr(_png(_make_qr(uri))))

    assert result.model_dump() == {
        "pa": "merchant@upi",
        "pn": "Example Shop",
        "am": Decimal("19.50"),
        "cu": "INR",
        "mc": "0000",
    }


def test_decode_qr_with_transparent_background() -> None:
    payload = "QR payload"
    grayscale = _make_qr(payload)
    image = Image.new("RGBA", grayscale.size, "black")
    image.putalpha(Image.eval(grayscale, lambda pixel: 255 - pixel))

    assert decode_qr(_png(image)) == payload


def test_reports_an_image_without_a_qr_code() -> None:
    with pytest.raises(QRCodeNotFoundError, match="No decodable QR code"):
        decode_qr(_png(Image.new("RGB", (160, 160), "white")))


def test_requires_a_disambiguated_qr_code() -> None:
    first = _make_qr("https://example.org/First")
    second = _make_qr("https://example.org/Second")
    canvas = Image.new("L", (first.width + second.width + 40, max(first.height, second.height)), 255)
    canvas.paste(first, (0, 0))
    canvas.paste(second, (first.width + 40, 0))

    with pytest.raises(MultipleQRCodesError, match="single QR code"):
        decode_qr(_png(canvas))


def test_reports_native_qr_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_decode() -> None:
        raise cv2.error("Native diagnostic")

    monkeypatch.setattr(cv2, "QRCodeDetector", failed_decode)

    with pytest.raises(QRCodeError, match="OpenCV could not process"):
        decode_qr(_png(Image.new("RGB", (10, 10))))


@pytest.mark.parametrize("payload", ["some\x07text", "x" * (MAX_TEXT_LENGTH + 1)])
def test_rejects_invalid_decoded_qr_text(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    class Detector:
        def detectAndDecodeMulti(self, gray: np.ndarray) -> tuple:
            return True, (payload,), None, ()

    monkeypatch.setattr(cv2, "QRCodeDetector", Detector)

    with pytest.raises(QRCodeError, match="(control characters|length limit)"):
        decode_qr(_png(Image.new("RGB", (10, 10))))


@pytest.mark.parametrize("data", [b"", b"not an image", _png(Image.new("RGB", (10, 10)))[:24]])
def test_rejects_empty_and_corrupt_qr_images(data: bytes) -> None:
    with pytest.raises(QRCodeError, match="image"):
        decode_qr(data)


def test_image_byte_limit_is_checked_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_images, "MAX_IMAGE_BYTES", 4)

    with pytest.raises(QRCodeError, match="at most"):
        decode_qr(b"12345")


@pytest.mark.parametrize("limit", ["MAX_IMAGE_SIDE", "MAX_IMAGE_PIXELS"])
def test_excessive_image_dimensions_are_rejected(monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setattr(_images, limit, 8)

    with pytest.raises(QRCodeError, match="dimensions"):
        decode_qr(_png(Image.new("L", (10, 8))))


def test_static_upi_fields_are_optional() -> None:
    minimal = parse_upi_uri("UPI://PAY?pa=Sample@UPI")
    empty_optionals = parse_upi_uri("upi://pay?pa=demo@upi&pn=&am=&cu=&mc=")

    assert minimal.pa == "Sample@UPI"
    assert minimal.am is None
    assert (empty_optionals.pn, empty_optionals.am, empty_optionals.cu, empty_optionals.mc) == (
        None, None, None, None
    )


def test_parses_utf8_payee_name_and_ignores_unused_fields() -> None:
    data = parse_upi_uri("upi://pay?pa=shop@upi&pn=%E0%A4%9A%E0%A4%BE%E0%A4%AF&tn=Note&tn=Other")

    assert data.pn == "चाय"


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "upi://pay",
        "https://pay?pa=demo@upi",
        "upi://collect?pa=demo@upi",
        "upi://pay:443?pa=demo@upi",
        "upi://user@pay?pa=demo@upi",
        "upi://pay/other?pa=demo@upi",
        "upi://pay?pa=demo@upi#",
        "upi://pay?pa=demo@upi\n",
        "upi://pay?pn=Unencoded Name&pa=demo@upi",
        "upi://pay?pn=MissingPayee",
        "upi://pay?pa=demo",
        "upi://pay?pa=demo@upi&pa=other@upi",
        "upi://pay?pa=demo@upi&%70a=other@upi",
        "upi://pay?pa=demo@upi&pn=%C3%28",
        "upi://pay?pa=demo@upi&pn=%2x",
        "upi://pay?pa=demo@upi&badField",
        "upi://pay?pa=demo@upi&am=NaN",
        "upi://pay?pa=demo@upi&am=%2B2",
        "upi://pay?pa=demo@upi&am=-1.00",
        "upi://pay?pa=demo@upi&am=2e2",
        "upi://pay?pa=demo@upi&am=1.001",
        "upi://pay?pa=demo@upi&am=12345678901234567",
        "upi://pay?pa=demo@upi&cu=USD",
        "upi://pay?pa=demo@upi&mc=12345",
        "upi://pay?pa=demo@upi&" + "&".join(f"k{i}=x" for i in range(32)),
        "upi://pay?pa=demo@upi&tn=" + "a" * 4100,
    ],
)
def test_rejects_malformed_upi_uris(uri: str) -> None:
    with pytest.raises(InvalidUPIPayloadError):
        parse_upi_uri(uri)


def test_upi_parser_error_does_not_echo_sensitive_fields() -> None:
    with pytest.raises(InvalidUPIPayloadError) as caught:
        parse_upi_uri("upi://pay?pa=private@upi&pn=PrivateName&mc=invalid")

    assert "private@upi" not in str(caught.value)
    assert "PrivateName" not in str(caught.value)
