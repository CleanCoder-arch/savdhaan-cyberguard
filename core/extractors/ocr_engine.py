"""Extract reviewable screenshot text with the local Tesseract executable."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from statistics import fmean

from core.extractors._images import ImageInputError, load_image


_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
)
_LANGUAGE = re.compile(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*")


class OCRError(RuntimeError):
    """OCR could not produce a result."""


class OCRInputError(OCRError):
    """The input is not a supported image."""


class OCREngineUnavailable(OCRError):
    """Tesseract was not found or could not be executed."""


class OCRTimeoutError(OCRError):
    """Tesseract exceeded its configured timeout."""


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Unreviewed text and the mean word confidence on a 0–100 scale."""

    text: str = field(repr=False)
    confidence: float | None
    requires_review: bool = field(default=True, init=False)


def _find_tesseract() -> str | None:
    executable = shutil.which("tesseract")
    if executable and Path(executable).is_absolute():
        return executable
    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            if directory := os.environ.get(variable):
                candidate = Path(directory) / "Tesseract-OCR" / "tesseract.exe"
                if candidate.is_file():
                    return str(candidate)
    return None


def _parse_tsv(output: bytes) -> OCRResult:
    """Reconstruct the reading order from Tesseract's word-level TSV rows."""
    try:
        rows = output.decode("utf-8").split("\n")
        if rows[0].rstrip("\r") != _TSV_HEADER:
            raise ValueError

        lines: dict[tuple[str, ...], list[str]] = {}
        confidences: list[float] = []
        for row in rows[1:]:
            if not row.strip():
                continue
            fields = row.rstrip("\r").split("\t", maxsplit=11)
            if len(fields) != 12 or fields[0] not in {"1", "2", "3", "4", "5"}:
                raise ValueError
            if fields[0] != "5":
                continue

            word = " ".join(fields[11].replace("\x00", "").split())
            if not word:
                continue

            confidence = float(fields[10])
            if confidence != -1 and not 0 <= confidence <= 100:
                raise ValueError
            lines.setdefault(tuple(fields[1:5]), []).append(word)
            if confidence >= 0:
                confidences.append(confidence)
    except ValueError:
        raise OCRError("Tesseract returned invalid OCR data.") from None

    text_lines: list[str] = []
    previous_paragraph = None
    for line_id, words in lines.items():
        paragraph = line_id[:3]
        if previous_paragraph is not None and paragraph != previous_paragraph:
            text_lines.append("")
        text_lines.append(" ".join(words))
        previous_paragraph = paragraph

    return OCRResult(
        text="\n".join(text_lines),
        confidence=fmean(confidences) if confidences else None,
    )


def extract_ocr(
    image_bytes: bytes,
    *,
    language: str = "eng",
    timeout: float = 15.0,
    executable: str | os.PathLike[str] | None = None,
) -> OCRResult:
    """Run Tesseract over pipes and return text for a user to review."""
    if not isinstance(language, str) or len(language) > 128 or not _LANGUAGE.fullmatch(language):
        raise ValueError("Use Tesseract language codes such as eng or eng+hin.")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= 120
    ):
        raise ValueError("The OCR timeout must be greater than 0 and at most 120 seconds.")

    try:
        image = load_image(image_bytes)
    except ImageInputError as exc:
        raise OCRInputError(str(exc)) from None

    command = os.fspath(executable) if executable is not None else _find_tesseract()
    if not command:
        raise OCREngineUnavailable("Install Tesseract or configure its executable path.")

    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        png = buffer.getvalue()

    try:
        result = subprocess.run(
            [command, "stdin", "stdout", "-l", language, "--psm", "3", "tsv"],
            input=png,
            capture_output=True,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise OCRTimeoutError("Tesseract exceeded the OCR timeout.") from None
    except OSError:
        raise OCREngineUnavailable("The configured Tesseract executable could not run.") from None

    if b"Failed loading language" in result.stderr:
        raise OCRError("Tesseract could not load the requested language data.")
    if result.returncode != 0:
        raise OCRError("Tesseract could not complete OCR. Check the configured language data.")
    return _parse_tsv(result.stdout)
