import subprocess
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from core.extractors import _images, ocr_engine
from core.extractors.ocr_engine import (
    OCREngineUnavailable,
    OCRError,
    OCRInputError,
    OCRTimeoutError,
    extract_ocr,
)


_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
)


def _as_png(image: Image.Image) -> bytes:
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@pytest.fixture
def screenshot() -> bytes:
    return _as_png(Image.new("RGB", (120, 100), "white"))


def _word(text: str, confidence: str, *, paragraph: int = 1, line: int = 1) -> str:
    return f"5\t1\t1\t{paragraph}\t{line}\t1\t0\t0\t10\t10\t{confidence}\t{text}"


def _output(*rows: str) -> bytes:
    return ("\r\n".join((_HEADER, *rows)) + "\r\n").encode()


def test_ocr_uses_pipes_and_recovers_text_and_confidence(
    screenshot: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsv = _output(
        _word('Verify "UPI"', "80"),
        _word("  विवरण\tPay\x00  ", "90"),
        _word("Second line", "-1", line=2),
        _word("New paragraph", "100", paragraph=2),
    )
    executable = Path("Tesseract OCR") / "tesseract"

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        assert command == [str(executable), "stdin", "stdout", "-l", "eng+hin", "--psm", "3", "tsv"]
        assert options["capture_output"] is True
        assert options["timeout"] == 2.5
        assert "shell" not in options
        image_bytes = options["input"]
        assert isinstance(image_bytes, bytes)
        with Image.open(BytesIO(image_bytes)) as child_input:
            assert child_input.size == (120, 100)
            assert child_input.format == "PNG"
        return subprocess.CompletedProcess(command, 0, stdout=tsv, stderr=b"")

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    result = extract_ocr(screenshot, language="eng+hin", timeout=2.5, executable=executable)

    assert result.text == 'Verify "UPI" विवरण Pay\nSecond line\n\nNew paragraph'
    assert result.confidence == 90
    assert result.requires_review is True
    assert "Verify" not in repr(result)


@pytest.mark.parametrize("row", ["", _word("   ", "-1"), "1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t"])
def test_blank_ocr_result_has_no_confidence(
    screenshot: bytes, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=_output(row), stderr=b"")

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    result = extract_ocr(screenshot, executable="tesseract")

    assert result.text == ""
    assert result.confidence is None


def test_reports_unavailable_tesseract(screenshot: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_engine, "_find_tesseract", lambda: None)

    with pytest.raises(OCREngineUnavailable, match="Install Tesseract"):
        extract_ocr(screenshot)


def test_reports_tesseract_startup_failure(screenshot: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*args: object, **options: object) -> None:
        raise FileNotFoundError("private configured path")

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    with pytest.raises(OCREngineUnavailable, match="could not run") as caught:
        extract_ocr(screenshot, executable="tesseract")

    assert "private" not in str(caught.value)


def test_reports_ocr_timeout(screenshot: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **options: object) -> None:
        raise subprocess.TimeoutExpired(command, 0.5, output=b"Private message")

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    with pytest.raises(OCRTimeoutError, match="OCR timeout") as caught:
        extract_ocr(screenshot, timeout=0.5, executable="tesseract")

    assert "Private message" not in str(caught.value)


def test_reports_nonzero_exit_without_echoing_stderr(
    screenshot: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"Private diagnostic text")

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    with pytest.raises(OCRError, match="could not complete OCR") as caught:
        extract_ocr(screenshot, executable="tesseract")

    assert "Private diagnostic text" not in str(caught.value)


def test_reports_unavailable_language_even_if_tesseract_returns_partial_text(
    screenshot: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command, 0, stdout=_output(_word("Private text", "80")), stderr=b"Failed loading language 'hin'"
        )

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    with pytest.raises(OCRError, match="requested language data"):
        extract_ocr(screenshot, language="eng+hin", executable="tesseract")


@pytest.mark.parametrize(
    "output",
    [b"garbage", b"\xff", _output("5\tshort"), _output(_word("word", "NaN")), _output(_word("word", "101"))],
)
def test_reports_corrupt_ocr_output(
    screenshot: bytes, monkeypatch: pytest.MonkeyPatch, output: bytes
) -> None:
    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(ocr_engine.subprocess, "run", run)

    with pytest.raises(OCRError, match="invalid OCR data"):
        extract_ocr(screenshot, executable="tesseract")


@pytest.mark.parametrize("language", ["", "eng;ignored", "eng++hin", "../eng", "x" * 129])
def test_rejects_malformed_language_codes(screenshot: bytes, language: str) -> None:
    with pytest.raises(ValueError, match="language codes"):
        extract_ocr(screenshot, language=language)


@pytest.mark.parametrize("timeout", [0, -1, 121, float("inf"), float("nan"), True])
def test_rejects_unbounded_timeouts(screenshot: bytes, timeout: float) -> None:
    with pytest.raises(ValueError, match="OCR timeout"):
        extract_ocr(screenshot, timeout=timeout)


def test_invalid_image_fails_before_launching_ocr() -> None:
    with pytest.raises(OCRInputError, match="could not be read"):
        extract_ocr(b"invalid image", executable="no-such-ocr-binary")


def test_rejects_animated_images() -> None:
    with BytesIO() as buffer:
        start = Image.new("RGB", (10, 10), "white")
        frame = Image.new("RGB", (10, 10), "black")
        start.save(buffer, format="GIF", save_all=True, append_images=[frame])
        animation = buffer.getvalue()

    with pytest.raises(OCRInputError, match="single-frame"):
        extract_ocr(animation)


def test_load_image_applies_exif_orientation() -> None:
    with BytesIO() as buffer:
        image = Image.new("RGB", (6, 12), "white")
        exif = image.getexif()
        exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)

        assert _images.load_image(buffer.getvalue()).size == (12, 6)


def test_load_image_composites_palette_transparency() -> None:
    image = Image.new("P", (2, 1))
    image.putpalette([0, 0, 0, 255, 0, 0])
    image.putdata([0, 1])
    image.info["transparency"] = 0

    loaded = _images.load_image(_as_png(image))

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((1, 0)) == (255, 0, 0)


@pytest.mark.parametrize("data", ["image.png", bytearray(b"image"), memoryview(b"image"), None])
def test_load_image_rejects_non_bytes(data: object) -> None:
    with pytest.raises(_images.ImageInputError, match="Provide a nonempty image"):
        _images.load_image(data)


def test_ocr_runs_with_local_tesseract_when_available() -> None:
    if ocr_engine._find_tesseract() is None:
        pytest.skip("Tesseract executable is unavailable")
    image = Image.new("RGB", (600, 150), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 30), "SCAM ALERT", font=ImageFont.load_default(size=58), fill="black")

    result = extract_ocr(_as_png(image))

    assert result.text == "SCAM ALERT"
    assert result.confidence is not None
    assert 0 <= result.confidence <= 100
