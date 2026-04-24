"""
preprocessing.py
Gmail Local LLM Organizer — Stage 2: Preprocessing

Transforms raw Gmail API message payloads into clean, token-budgeted
dicts ready for LLM inference.

Pipeline (in order):
  1. Decode Gmail API payload — robust recursive MIME traversal
  2. Content selection: text/plain preferred; text/html converted via
     BeautifulSoup when plain is absent or empty
  3. Conservative cleaning (whitespace only — no aggressive filtering)
  4. Remove quoted reply chains
  5. Truncate body to ~2 000 chars (≈500 tokens)
  6. Normalize whitespace
  7. Fallback hierarchy: snippet → subject → subject+snippet (guarantees
     non-empty body)

Observability: every email emits a structured log line via the standard
``logging`` module so body-loss can be tracked in aggregate.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from email import message_from_bytes, message_from_string
from email.message import Message
from typing import Optional

try:
    from bs4 import BeautifulSoup  # preferred HTML parser
    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmailRecord:
    """Structured representation of a preprocessed email."""
    message_id: str
    subject: str
    sender: str           # display name + address, e.g. "Alice <alice@example.com>"
    date: str             # raw Date header value
    body: str             # cleaned, truncated body
    attachment_names: list[str] = field(default_factory=list)
    # Observability extras (not sent to LLM)
    body_source: str = field(default="extracted")   # extracted | snippet | subject | subject+snippet
    fallback_used: bool = field(default=False)


# ---------------------------------------------------------------------------
# HTML → plain text conversion
# ---------------------------------------------------------------------------

def _strip_html_bs4(html: str) -> str:
    """Extract visible text using BeautifulSoup (preferred)."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove invisible/non-content tags entirely.
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _strip_html_stdlib(html: str) -> str:
    """
    Fallback HTML extractor using the stdlib html.parser.
    Handles block-level tags for paragraph preservation.
    """
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3",
                  "h4", "h5", "h6", "blockquote", "article", "section"}
        _SKIP  = {"script", "style", "head", "meta", "link"}

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self._buf: list[str] = []
            self._skip_depth = 0

        def handle_starttag(self, tag: str, attrs) -> None:
            if tag in self._SKIP:
                self._skip_depth += 1
            elif tag in self._BLOCK:
                self._buf.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in self._SKIP:
                self._skip_depth = max(0, self._skip_depth - 1)

        def handle_data(self, data: str) -> None:
            if self._skip_depth == 0:
                self._buf.append(data)

        def result(self) -> str:
            return "".join(self._buf)

    p = _Stripper()
    try:
        p.feed(html)
        return p.result()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def strip_html(html: str) -> str:
    """Return plain text extracted from an HTML string."""
    if _BS4_AVAILABLE:
        return _strip_html_bs4(html)
    return _strip_html_stdlib(html)


# ---------------------------------------------------------------------------
# Quoted-reply removal
# ---------------------------------------------------------------------------

_QUOTE_LINE_RE    = re.compile(r"^\s*>")
_WROTE_HEADER_RE  = re.compile(
    r"^\s*On\s.{10,120}wrote:\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FORWARDED_RE     = re.compile(
    r"^\s*-{3,}\s*Forwarded [Mm]essage\s*-{3,}",
    re.IGNORECASE,
)


def remove_quoted_replies(text: str) -> str:
    """
    Strip quoted reply blocks from a plain-text email body.

    Handles:
    - Lines starting with '>'
    - 'On <date> <name> wrote:' headers (and everything after)
    - '--- Forwarded message ---' blocks
    """
    lines   = text.splitlines()
    cleaned: list[str] = []

    for line in lines:
        if _WROTE_HEADER_RE.match(line) or _FORWARDED_RE.match(line):
            break
        if _QUOTE_LINE_RE.match(line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Token-budget truncation
# ---------------------------------------------------------------------------

_CHAR_BUDGET = 2_000   # ≈ 500 tokens @ 4 chars/token


def truncate_to_token_budget(text: str, char_budget: int = _CHAR_BUDGET) -> str:
    """Truncate *text* to at most *char_budget* characters at a word boundary."""
    if len(text) <= char_budget:
        return text
    truncated = text[:char_budget]
    last_space = truncated.rfind(" ")
    if last_space > char_budget * 0.8:
        truncated = truncated[:last_space]
    return truncated + " […]"

# ---------------------------------------------------------------------------
# Unicode cleanup (invisible + low-signal characters)
# ---------------------------------------------------------------------------

# Common problematic Unicode characters in emails
_INVISIBLE_CHARS_RE = re.compile(
    r"[\u200B-\u200D\uFEFF]"  # zero-width space, joiners, BOM
)

_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"  # non-printable ASCII control chars
)

_NON_BREAKING_SPACE_RE = re.compile(r"\xa0")


def clean_unicode(text: str) -> str:
    """
    Remove invisible and low-information Unicode characters
    without affecting readable content.
    """
    text = _INVISIBLE_CHARS_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _NON_BREAKING_SPACE_RE.sub(" ", text)
    return text

# ---------------------------------------------------------------------------
# Whitespace normalization  (conservative — no content removal)
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines to one; strip trailing spaces and edges."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# MIME decoding helpers
# ---------------------------------------------------------------------------

def _decode_payload(part: Message) -> str:
    """Decode a MIME part payload to a Python str."""
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


# ---------------------------------------------------------------------------
# Gmail API payload → text  (robust recursive traversal)
# ---------------------------------------------------------------------------

def _collect_parts(
    gmail_payload: dict,
    plain_parts: list[str],
    html_parts: list[str],
    attachment_names: list[str],
    depth: int = 0,
) -> None:
    """
    Recursively walk a Gmail API payload dict, collecting:
      - decoded text/plain bodies → plain_parts
      - decoded text/html bodies → html_parts
      - attachment filenames     → attachment_names

    Handles arbitrarily nested multipart/alternative, multipart/mixed, etc.
    """
    mime_type: str  = gmail_payload.get("mimeType", "").lower()
    body: dict      = gmail_payload.get("body", {})
    parts: list     = gmail_payload.get("parts", [])
    headers: list   = gmail_payload.get("headers", [])

    header_map = {h["name"].lower(): h["value"] for h in headers}
    disposition = header_map.get("content-disposition", "")

    logger.debug(
        "  %s[depth=%d] mimeType=%r  parts=%d  body.size=%s",
        "  " * depth, depth, mime_type, len(parts),
        body.get("size", "?"),
    )

    # --- Attachment ---------------------------------------------------------
    if "attachment" in disposition:
        filename = header_map.get("content-disposition", "")
        # Try to pull the filename from Content-Disposition value
        fname_match = re.search(r'filename=["\']?([^"\';\s]+)', filename, re.IGNORECASE)
        if fname_match:
            attachment_names.append(fname_match.group(1))
        elif mime_type not in ("text/plain", "text/html", ""):
            attachment_names.append(f"attachment.{mime_type.split('/')[-1]}")
        return

    # --- Recurse into sub-parts first  (handles multipart/*) ----------------
    if parts:
        for sub in parts:
            _collect_parts(sub, plain_parts, html_parts, attachment_names, depth + 1)
        return

    # --- Leaf part: decode base64url body -----------------------------------
    data = body.get("data", "")
    if not data:
        return

    try:
        decoded = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Base64 decode error at depth %d: %s", depth, exc)
        return

    if mime_type == "text/plain":
        plain_parts.append(decoded)
    elif mime_type == "text/html":
        html_parts.append(decoded)
    # Other leaf types (image/*, application/*) are silently ignored.


def _extract_from_gmail_payload(
    payload: dict,
) -> tuple[str, list[str], list[str]]:
    """
    Extract body text and metadata from a Gmail API payload dict.

    Returns
    -------
    (raw_body, plain_parts_log, attachment_names)
    Where raw_body is the best plain-text candidate before cleaning.
    """
    plain_parts: list[str] = []
    html_parts:  list[str] = []
    attachment_names: list[str] = []

    _collect_parts(payload, plain_parts, html_parts, attachment_names)

    logger.debug(
        "MIME scan: %d plain part(s), %d html part(s), %d attachment(s)",
        len(plain_parts), len(html_parts), len(attachment_names),
    )

    # --- Content selection: prefer text/plain; fall back to text/html -------
    # Filter out empty / whitespace-only parts first.
    non_empty_plain = [p for p in plain_parts if p.strip()]
    non_empty_html  = [p for p in html_parts  if p.strip()]

    if non_empty_plain:
        raw_plain = "\n\n".join(non_empty_plain)
        # Some senders mis-label HTML as text/plain; detect and convert.
        if re.search(r"<[a-zA-Z][^>]*>", raw_plain):
            raw_plain = strip_html(raw_plain)
        chosen_type = "text/plain"
        body_text = raw_plain
    elif non_empty_html:
        body_text = strip_html("\n\n".join(non_empty_html))
        chosen_type = "text/html→text"
    else:
        body_text = ""
        chosen_type = "none"

    logger.debug("Chosen content type: %s  raw body length: %d", chosen_type, len(body_text))
    return body_text, attachment_names # type: ignore


# ---------------------------------------------------------------------------
# IMAP / raw-bytes MIME extraction  (reuses stdlib email library)
# ---------------------------------------------------------------------------

def extract_text_and_attachments(
    raw: bytes | str,
) -> tuple[str, list[str]]:
    """
    Walk a raw RFC 2822 MIME message and return:
      - body_text: the best plain-text representation
      - attachment_names: list of filenames for all attachments
    """
    if isinstance(raw, bytes):
        msg: Message = message_from_bytes(raw)
    else:
        msg: Message = message_from_string(str(raw))

    plain_parts: list[str] = []
    html_parts:  list[str] = []
    attachment_names: list[str] = []

    for part in msg.walk():
        disposition  = part.get("Content-Disposition", "")
        content_type = part.get_content_type()

        if "attachment" in disposition:
            filename = part.get_filename()
            if filename:
                attachment_names.append(filename)
            continue

        if content_type == "text/plain":
            decoded = _decode_payload(part)
            if decoded.strip():
                plain_parts.append(decoded)
        elif content_type == "text/html":
            decoded = _decode_payload(part)
            if decoded.strip():
                html_parts.append(decoded)

    if plain_parts:
        raw_plain = "\n\n".join(plain_parts)
        if re.search(r"<[a-zA-Z][^>]*>", raw_plain):
            raw_plain = strip_html(raw_plain)
        body_text = raw_plain
    elif html_parts:
        body_text = strip_html("\n\n".join(html_parts))
    else:
        body_text = ""

    return body_text, attachment_names


# ---------------------------------------------------------------------------
# Fallback hierarchy
# ---------------------------------------------------------------------------

_MIN_BODY_LEN = 20  # chars; bodies shorter than this trigger fallback


def _apply_fallback(
    body: str,
    subject: str,
    snippet: str,
) -> tuple[str, bool, str]:
    """
    Guarantee a non-empty body using the fallback hierarchy:
      1. extracted body (if long enough)
      2. Gmail snippet
      3. subject line
      4. subject + snippet

    Returns (final_body, fallback_used, source_label).
    """
    if len(body.strip()) >= _MIN_BODY_LEN:
        return body, False, "extracted"

    if snippet and len(snippet.strip()) >= _MIN_BODY_LEN:
        logger.info("Fallback: using Gmail snippet (extracted body too short)")
        return snippet.strip(), True, "snippet"

    if subject and subject != "(no subject)":
        combined = f"{subject}\n{snippet}".strip() if snippet else subject.strip()
        logger.info("Fallback: using subject (+ snippet) as body")
        return combined, True, "subject+snippet" if snippet else "subject"

    # Last resort: return whatever we have even if it's very short.
    logger.warning("Fallback exhausted — body may be empty or very short")
    return body.strip(), True, "exhausted"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_gmail_message(gmail_message: dict) -> EmailRecord:
    """
    Convert a raw Gmail API ``messages.get()`` response dict into a clean
    :class:`EmailRecord` ready for LLM inference.

    Parameters
    ----------
    gmail_message:
        The full message dict returned by the Gmail API, including 'id',
        'payload', 'snippet', and top-level headers.

    Returns
    -------
    EmailRecord
        Cleaned, truncated email with guaranteed non-empty body.
    """
    message_id: str = gmail_message.get("id", "")
    payload:    dict = gmail_message.get("payload", {})
    snippet:    str  = gmail_message.get("snippet", "")
    headers:    list = payload.get("headers", [])

    header_map = {h["name"].lower(): h["value"] for h in headers}
    subject: str = header_map.get("subject", "(no subject)")
    sender:  str = header_map.get("from",    "(unknown sender)")
    date:    str = header_map.get("date",    "")

    logger.debug(
        "Preprocessing message_id=%s  subject=%r  snippet_len=%d",
        message_id, subject[:60], len(snippet),
    )

    # --- Extract body -------------------------------------------------------
    raw_body, attachment_names = _extract_from_gmail_payload(payload) # type: ignore
    logger.debug("Body length before cleaning: %d", len(raw_body))

    # --- Clean pipeline -----------------------------------------------------
    body = remove_quoted_replies(raw_body)
    body = truncate_to_token_budget(body)
    body = normalize_whitespace(body)
    logger.debug("Body length after cleaning: %d", len(body))

    # --- Fallback -----------------------------------------------------------
    body, fallback_used, body_source = _apply_fallback(body, subject, snippet)

    logger.info(
        "message_id=%s  body_length=%d  fallback=%s  source=%s",
        message_id, len(body), fallback_used, body_source,
    )

    return EmailRecord(
        message_id=message_id,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachment_names=attachment_names,
        body_source=body_source,
        fallback_used=fallback_used,
    )


def preprocess_raw_bytes(
    message_id: str,
    raw: bytes,
    snippet: str = "",
) -> EmailRecord:
    """
    Preprocess an email delivered as raw RFC 2822 bytes (e.g. from IMAP).

    Parameters
    ----------
    message_id:
        An identifier string for the message (e.g. IMAP UID).
    raw:
        Raw email bytes.
    snippet:
        Optional short preview string to use as fallback body.

    Returns
    -------
    EmailRecord
    """
    msg = message_from_bytes(raw)

    subject: str = msg.get("Subject", "(no subject)")
    sender:  str = msg.get("From",    "(unknown sender)")
    date:    str = msg.get("Date",    "")

    logger.debug(
        "Preprocessing raw bytes  message_id=%s  subject=%r",
        message_id, subject[:60],
    )

    raw_body, attachment_names = extract_text_and_attachments(raw)
    logger.debug("Body length before cleaning: %d", len(raw_body))

    body = remove_quoted_replies(raw_body)
    body = truncate_to_token_budget(body)
    body = normalize_whitespace(body)
    logger.debug("Body length after cleaning: %d", len(body))

    body, fallback_used, body_source = _apply_fallback(body, subject, snippet)

    logger.info(
        "message_id=%s  body_length=%d  fallback=%s  source=%s",
        message_id, len(body), fallback_used, body_source,
    )

    return EmailRecord(
        message_id=message_id,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachment_names=attachment_names,
        body_source=body_source,
        fallback_used=fallback_used,
    )
