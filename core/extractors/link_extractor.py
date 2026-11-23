"""Extract absolute web links from untrusted text without opening them."""

import re

from core.validators import MAX_TEXT_LENGTH, validate_text, validate_url

_WEB_LINK = re.compile(
    r"""<(?P<angle_link>https?://[^\s<>"]+)>"""
    r'''|"(?P<double_quoted_link>https?://[^\s<>"]+)"'''
    r"""|(?<![\w+.-])(?P<bare_link>https?://[^\s<>"'’‘“”]+)""",
    re.IGNORECASE,
)
_CLOSING_DELIMITERS = {")": "(", "]": "[", "}": "{"}
_CLOSING_QUOTES = {"'": "'", "‘": "’", "“": "”"}
_PROSE_PUNCTUATION = ".,;!:?"


def _trim_prose_punctuation(candidate: str) -> str:
    unmatched = {
        closer: candidate.count(closer) - candidate.count(opener)
        for closer, opener in _CLOSING_DELIMITERS.items()
    }
    end = len(candidate)
    while end:
        character = candidate[end - 1]
        if character not in _PROSE_PUNCTUATION:
            if unmatched.get(character, 0) <= 0:
                break
            unmatched[character] -= 1
        end -= 1
    return candidate[:end]


def extract_links(text: str) -> tuple[str, ...]:
    """Return valid links in first-seen order, preserving path and query case."""
    if isinstance(text, str) and len(text) <= MAX_TEXT_LENGTH and not text.strip():
        return ()
    validate_text(text)
    links: dict[str, None] = {}
    for match in _WEB_LINK.finditer(text):
        candidate = match.group("angle_link") or match.group("double_quoted_link")
        if candidate is None:
            start, end = match.span("bare_link")
            candidate = match.group("bare_link")
            if start == 0 or end == len(text) or _CLOSING_QUOTES.get(text[start - 1]) != text[end]:
                candidate = _trim_prose_punctuation(candidate)
        try:
            link = validate_url(candidate)
        except ValueError:
            # Invalid URL-shaped text is outside the extractor's result contract.
            continue
        links[link] = None
    return tuple(links)
