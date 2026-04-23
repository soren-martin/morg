"""
preprocessing.py
Gmail Local LLM Organizer — Stage 2: Preprocessing

Transforms raw Gmail API message payloads into clean, token-budgeted
dicts ready for LLM inference.

Pipeline (in order):
  1. Decode MIME multipart — extract text/plain; collect attachment metadata
  2. Strip HTML tags from any text/html fallback parts
  3. Remove quoted reply chains
  4. Truncate body to 500 tokens (~375 words)
  5. Normalize whitespace
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from email import message_from_bytes, message_from_string
from email.message import Message
from html.parser import HTMLParser
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmailRecord:
    """Structured representation of a preprocessed email."""
    message_id: str
    subject: str
    sender: str          # display name + address, e.g. "Alice <alice@example.com>"
    date: str            # raw Date header value
    body: str            # cleaned, truncated body
    attachment_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text converter that preserves whitespace structure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        # Insert newlines around block-level elements so paragraphs survive.
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3",
                   "h4", "h5", "h6", "blockquote"}:
            self._parts.append("\n")

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(html: str) -> str:
    """Return plain text extracted from an HTML string."""
    parser = _HTMLStripper()
    try:
        parser.feed(html)
    except Exception:
        # Malformed HTML — fall back to a naive tag-strip.
        return re.sub(r"<[^>]+>", " ", html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Quoted-reply removal
# ---------------------------------------------------------------------------

# Lines that indicate the start of a quoted reply block.
_QUOTE_LINE_RE = re.compile(r"^\s*>")          # "> quoted text"
_WROTE_HEADER_RE = re.compile(
    r"^\s*On\s.{10,120}wrote:\s*$",             # "On Mon, 1 Jan 2024 Alice wrote:"
    re.IGNORECASE | re.DOTALL,
)
_FORWARDED_RE = re.compile(
    r"^\s*-{3,}\s*Forwarded [Mm]essage\s*-{3,}",
    re.IGNORECASE,
)


def remove_quoted_replies(text: str) -> str:
    """
    Strip quoted reply blocks from plain text email body.

    Handles:
    - Lines starting with '>'
    - 'On <date> <name> wrote:' separator lines and everything after
    - '--- Forwarded message ---' blocks
    """
    lines = text.splitlines()
    cleaned: list[str] = []

    for line in lines:
        # Once we hit a "wrote:" header or forwarded block, drop the rest.
        if _WROTE_HEADER_RE.match(line) or _FORWARDED_RE.match(line):
            break
        if _QUOTE_LINE_RE.match(line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Token-budget truncation
# ---------------------------------------------------------------------------

# 500 tokens * ~4 chars/token = 2000 chars. Character truncation is more
# reliable than word-splitting on messy email text that may contain long
# unbroken strings (URLs, base64 fragments, MIME headers).
_CHAR_BUDGET = 2000


def truncate_to_token_budget(text: str, char_budget: int = _CHAR_BUDGET) -> str:
    """
    Truncate *text* to at most *char_budget* characters.
    This reliably caps prompt size regardless of token/word ambiguity.
    """
    if len(text) <= char_budget:
        return text
    # Try to break at a word boundary rather than mid-word.
    truncated = text[:char_budget]
    last_space = truncated.rfind(" ")
    if last_space > char_budget * 0.8:  # only snap back if space is close
        truncated = truncated[:last_space]
    return truncated + " […]"


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines to a single blank line; strip edges."""
    # Collapse 3+ consecutive newlines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# MIME decoding
# ---------------------------------------------------------------------------

def _decode_payload(part: Message) -> str:
    """Decode a MIME part payload to a Python str."""
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def extract_text_and_attachments(
    raw: bytes | str,
) -> tuple[str, list[str]]:
    """
    Walk a MIME message tree and return:
      - body_text: the best plain-text representation of the message body
      - attachment_names: list of filenames for all attachments
    """
    if isinstance(raw, bytes):
        msg: Message = message_from_bytes(raw)
    else:
        msg: Message = message_from_string(str(raw))

    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachment_names: list[str] = []

    for part in msg.walk():
        disposition = part.get("Content-Disposition", "")
        content_type = part.get_content_type()

        if "attachment" in disposition:
            filename = part.get_filename()
            if filename:
                attachment_names.append(filename)
            continue

        if content_type == "text/plain":
            plain_parts.append(_decode_payload(part))
        elif content_type == "text/html":
            html_parts.append(_decode_payload(part))

    if plain_parts:
        raw_plain = "\n\n".join(plain_parts)
        # Some senders label HTML content as text/plain. Strip tags if present.
        if re.search(r"<[a-zA-Z][^>]*>", raw_plain):
            raw_plain = strip_html(raw_plain)
        body_text = raw_plain
    elif html_parts:
        body_text = strip_html("\n\n".join(html_parts))
    else:
        body_text = ""

    return body_text, attachment_names

# ---------------------------------------------------------------------------
# Gmail API payload decoder
# ---------------------------------------------------------------------------

def _gmail_part_to_mime(gmail_payload: dict) -> str:
    """
    Recursively reconstruct a raw MIME string from a Gmail API message
    payload dict (the 'payload' key of a messages.get() response).

    Gmail base64url-encodes each part's body individually, which the
    standard email library cannot parse directly, so we decode each part
    and stitch the text back together.
    """
    mime_type: str = gmail_payload.get("mimeType", "")
    headers: list[dict] = gmail_payload.get("headers", [])
    body: dict = gmail_payload.get("body", {})
    parts: list[dict] = gmail_payload.get("parts", [])

    header_block = "\n".join(
        f"{h['name']}: {h['value']}" for h in headers
    )

    if parts:
        # Multipart: recurse and join.
        children = "\n\n".join(_gmail_part_to_mime(p) for p in parts)
        return f"{header_block}\n\n{children}"

    # Leaf part — decode the base64url body.
    data = body.get("data", "")
    if data:
        decoded = base64.urlsafe_b64decode(data + "==").decode(
            "utf-8", errors="replace"
        )
    else:
        decoded = ""

    return f"{header_block}\n\n{decoded}"


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
        'payload', and top-level headers.

    Returns
    -------
    EmailRecord
        Cleaned, truncated email suitable for the LLM prompt.
    """
    message_id: str = gmail_message.get("id", "")
    payload: dict = gmail_message.get("payload", {})
    headers: list[dict] = payload.get("headers", [])

    # --- Extract headers ---------------------------------------------------
    header_map = {h["name"].lower(): h["value"] for h in headers}
    subject: str = header_map.get("subject", "(no subject)")
    sender: str = header_map.get("from", "(unknown sender)")
    date: str = header_map.get("date", "")

    # --- Decode MIME body and attachments ----------------------------------
    raw_mime = _gmail_part_to_mime(payload)
    raw_body, attachment_names = extract_text_and_attachments(raw_mime)

    # --- Apply cleaning pipeline -------------------------------------------
    body = remove_quoted_replies(raw_body)
    body = truncate_to_token_budget(body)
    body = normalize_whitespace(body)

    return EmailRecord(
        message_id=message_id,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachment_names=attachment_names,
    )


def preprocess_raw_bytes(
    message_id: str,
    raw: bytes,
) -> EmailRecord:
    """
    Preprocess an email delivered as raw RFC 2822 bytes (e.g. from IMAP).

    Parameters
    ----------
    message_id:
        An identifier string for the message (e.g. IMAP UID).
    raw:
        Raw email bytes.

    Returns
    -------
    EmailRecord
    """
    msg = message_from_bytes(raw)

    subject: str = msg.get("Subject", "(no subject)")
    sender: str = msg.get("From", "(unknown sender)")
    date: str = msg.get("Date", "")

    raw_body, attachment_names = extract_text_and_attachments(raw)

    body = remove_quoted_replies(raw_body)
    body = truncate_to_token_budget(body)
    body = normalize_whitespace(body)

    return EmailRecord(
        message_id=message_id,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachment_names=attachment_names,
    )
