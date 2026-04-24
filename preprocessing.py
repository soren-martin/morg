from __future__ import annotations

import base64
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import html2text  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmailRecord:
    message_id: str
    subject: str
    sender: str
    date: str
    body: str
    attachment_names: list[str] = field(default_factory=list)
    body_source: str = field(default="extracted")
    fallback_used: bool = field(default=False)

# ---------------------------------------------------------------------------
# HTML → TEXT (improved)
# ---------------------------------------------------------------------------

_h = html2text.HTML2Text()
_h.ignore_links = False
_h.ignore_images = False
_h.images_to_alt = True
_h.ignore_emphasis = True
_h.body_width = 0


def strip_html(html: str) -> str:
    return _h.handle(html)


def extract_hidden_preview(html: str) -> str:
    matches = re.findall(
        r'<(?:span|div)[^>]*display\s*:\s*none[^>]*>(.*?)</(?:span|div)>',
        html,
        re.IGNORECASE | re.DOTALL
    )
    return " ".join(matches)


def clean_links(text: str) -> str:
    return re.sub(r"\(https?://[^\)]+\)", "", text)

# ---------------------------------------------------------------------------
# Reply + signature removal
# ---------------------------------------------------------------------------

REPLY_SPLIT_PATTERNS = [
    r"^\s*On .{0,200}?wrote:\s*$",
    r"^\s*From:\s.+",
    r"^\s*Sent:\s.+",
    r"^\s*To:\s.+",
    r"^\s*Subject:\s.+",
    r"^\s*-{2,}\s*Original Message\s*-{2,}",
    r"^\s*-{2,}\s*Forwarded message\s*-{2,}",
]

QUOTE_LINE_RE = re.compile(r"^\s*>+")

SIGNATURE_PATTERNS = [
    r"^\s*--\s*$",
    r"^\s*Best regards,?\s*$",
    r"^\s*Regards,?\s*$",
    r"^\s*Sincerely,?\s*$",
    r"^\s*Thanks,?\s*$",
    r"^\s*Sent from my iPhone",
    r"^\s*Sent from my .*",
]


def remove_quoted_replies(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        if any(re.match(p, line, re.IGNORECASE) for p in REPLY_SPLIT_PATTERNS):
            break
        if QUOTE_LINE_RE.match(line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def remove_signature(text: str) -> str:
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if any(re.match(p, line, re.IGNORECASE) for p in SIGNATURE_PATTERNS):
            return "\n".join(lines[:i])

    return text

# ---------------------------------------------------------------------------
# Unicode cleanup
# ---------------------------------------------------------------------------

_INVISIBLE_CHARS_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_NON_BREAKING_SPACE_RE = re.compile(r"\xa0")


def clean_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    text = _INVISIBLE_CHARS_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _NON_BREAKING_SPACE_RE.sub(" ", text)

    replacements = {
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "–": "-", "—": "-",
        "…": "...",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    return text

# ---------------------------------------------------------------------------
# Structure + whitespace
# ---------------------------------------------------------------------------

_CHAR_BUDGET = 2000


def preserve_structure(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"(?<=\n)([A-Z][A-Z\s]{5,})(?=\n)", r"\n\1\n", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def truncate_to_token_budget(text: str, char_budget: int = _CHAR_BUDGET) -> str:
    if len(text) <= char_budget:
        return text
    truncated = text[:char_budget]
    last_space = truncated.rfind(" ")
    if last_space > char_budget * 0.8:
        truncated = truncated[:last_space]
    return truncated + " […]"

# ---------------------------------------------------------------------------
# MIME traversal (improved)
# ---------------------------------------------------------------------------


def _collect_parts(payload, plain_parts, html_parts, attachments):
    mime_type = payload.get("mimeType", "").lower()
    body = payload.get("body", {})
    parts = payload.get("parts", [])
    headers = payload.get("headers", [])

    header_map = {h["name"].lower(): h["value"] for h in headers}
    disposition = header_map.get("content-disposition", "")

    filename = payload.get("filename")

    if "attachment" in disposition:
        if filename:
            attachments.append(filename)
        else:
            attachments.append("attachment")
        return

    if parts:
        for p in parts:
            _collect_parts(p, plain_parts, html_parts, attachments)
        return

    data = body.get("data")
    if not data:
        return

    try:
        decoded = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return

    if mime_type == "text/plain":
        plain_parts.append(decoded)
    elif mime_type == "text/html":
        html_parts.append(decoded)


def _extract_from_gmail_payload(payload):
    plain_parts, html_parts, attachments = [], [], []
    _collect_parts(payload, plain_parts, html_parts, attachments)

    if plain_parts and len(" ".join(plain_parts)) > 200:
        text = "\n\n".join(plain_parts)
        if re.search(r"<[a-zA-Z]", text):
            text = strip_html(text)
        return text, attachments

    if html_parts:
        html = "\n\n".join(html_parts)
        preview = extract_hidden_preview(html)
        text = strip_html(html)

        if preview:
            text = preview + "\n\n" + text

        return text, attachments

    return "", attachments

# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

_MIN_BODY_LEN = 20


def _apply_fallback(body, subject, snippet):
    if len(body.strip()) >= _MIN_BODY_LEN:
        return body, False, "extracted"

    if snippet and len(snippet.strip()) >= _MIN_BODY_LEN:
        return snippet.strip(), True, "snippet"

    if subject:
        combined = f"{subject}\n{snippet}" if snippet else subject
        return combined.strip(), True, "subject+snippet"

    return body.strip(), True, "exhausted"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_gmail_message(gmail_message: dict) -> EmailRecord:
    message_id = gmail_message.get("id", "")
    payload = gmail_message.get("payload", {})
    snippet = gmail_message.get("snippet", "")
    headers = payload.get("headers", [])

    header_map = {h["name"].lower(): h["value"] for h in headers}
    subject = header_map.get("subject", "(no subject)")
    sender = header_map.get("from", "(unknown sender)")
    date = header_map.get("date", "")

    raw_body, attachments = _extract_from_gmail_payload(payload)

    body = remove_quoted_replies(raw_body)
    body = remove_signature(body)

    body = clean_unicode(body)
    body = preserve_structure(body)
    body = clean_links(body)
    body = normalize_whitespace(body)
    body = truncate_to_token_budget(body)

    body, fallback_used, source = _apply_fallback(body, subject, snippet)

    return EmailRecord(
        message_id=message_id,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachment_names=attachments,
        body_source=source,
        fallback_used=fallback_used,
    )
