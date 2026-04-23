"""
llm_inference.py
Gmail Local LLM Organizer — Stage 3: LLM Inference

Sends preprocessed EmailRecord objects to a locally hosted Mistral 7B model
via the Ollama HTTP API and parses the structured JSON classification response.

Design spec (§4.3):
  - Model  : mistral:7b-instruct-q4_K_M served by Ollama
  - Output : strict JSON — { category, action_required, action_reason }
  - Throttle: configurable inter-request delay (default 2 s) to avoid
              thermal throttling on the 1360P under a 60W power envelope
  - Threads : Ollama num_thread set to 6 to favour P-cores without
              saturating efficiency cores needed for OS background tasks
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests

from preprocessing import EmailRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "mistral:7b-instruct-q4_K_M"

VALID_CATEGORIES = frozenset(
    {"correspondence", "advertising", "newsletter", "notification", "spam"}
)

Category = Literal["correspondence", "advertising", "newsletter", "notification", "spam"]

# System prompt — instructs the model to return only valid JSON, no preamble.
_SYSTEM_PROMPT = (
    "You are an email classifier. Return only valid JSON. No explanation."
)

# User prompt template — mirrors the schema shown in §4.3.
_USER_PROMPT_TEMPLATE = """\
Classify this email. Return a JSON object with these keys:
  - "category": one of [correspondence, advertising, newsletter, notification, spam]
  - "action_required": true or false
  - "action_reason": brief string if action_required is true, else null

  Category definitions:
  - correspondence: Direct personal or professional communication from a real person \
expecting a reply or acknowledgment; includes meeting requests, questions, project \
discussions, and job-related emails.
  - advertising: Promotional content from a brand or service trying to sell or upsell \
something; includes sales, deals, discount codes, product launches, and re-engagement \
campaigns.
  - newsletter: Regularly scheduled digest or editorial content the user subscribed to; \
includes industry roundups, blogs, and curated link collections — not a direct sales pitch.
  - notification: Automated system-generated alert about activity in an account or \
service; includes shipping updates, login alerts, password resets, receipts, and \
calendar invites.
  - spam: Unsolicited, deceptive, or malicious email the user did not request and has \
no legitimate relationship with; includes phishing, scams, and bulk junk.

Subject: {subject}
From: {sender}
Body: {body}{attachment_block}\
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """LLM classification output for a single email."""
    message_id:      str
    category:        Category
    action_required: bool
    action_reason:   Optional[str]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_user_prompt(record: EmailRecord) -> str:
    """Render the user-turn prompt from a preprocessed EmailRecord."""
    attachment_block = ""
    if record.attachment_names:
        names = ", ".join(record.attachment_names)
        attachment_block = f"\nAttachments: {names}"

    return _USER_PROMPT_TEMPLATE.format(
        subject=record.subject,
        sender=record.sender,
        body=record.body,
        attachment_block=attachment_block,
    )


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

def _call_ollama(
    user_prompt: str,
    *,
    model: str,
    base_url: str,
    num_thread: int,
    timeout: float,
) -> str:
    """
    POST a chat-completion request to the Ollama /api/chat endpoint and
    return the raw response text.

    Raises
    ------
    requests.HTTPError
        If Ollama returns a non-2xx status code.
    requests.Timeout
        If the request exceeds *timeout* seconds.
    """
    payload = {
        "model": model,
        "stream": False,
        "options": {
            # Limit thread count to favour P-cores; see §4.3 Throttling.
            "num_thread": num_thread,
        },
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    # Ollama chat response: { "message": { "content": "..." }, ... }
    return data["message"]["content"]


# ---------------------------------------------------------------------------
# JSON parsing & validation
# ---------------------------------------------------------------------------

def _parse_classification(raw: str, message_id: str) -> ClassificationResult:
    """
    Parse and validate the model's raw text response into a
    :class:`ClassificationResult`.

    The model is instructed to return bare JSON, but it occasionally wraps
    the output in a ```json … ``` fence — strip that if present.

    Raises
    ------
    ValueError
        If the text cannot be parsed as JSON or fails schema validation.
    """
    text = raw.strip()

    # Strip optional markdown code fence.
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```).
        text = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output: {raw!r}") from exc

    # --- Validate required keys -------------------------------------------
    missing = {"category", "action_required", "action_reason"} - obj.keys()
    if missing:
        raise ValueError(f"Missing keys in model response: {missing}")

    category: str = str(obj["category"]).lower()
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Unknown category {category!r}. Expected one of {sorted(VALID_CATEGORIES)}."
        )

    action_required = bool(obj["action_required"])
    action_reason   = obj["action_reason"] if action_required else None

    # Coerce a non-null action_reason to a plain string.
    if action_reason is not None:
        action_reason = str(action_reason).strip() or None

    return ClassificationResult(
        message_id=message_id,
        category=category,          # type: ignore[arg-type]
        action_required=action_required,
        action_reason=action_reason,
    )


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

class EmailClassifier:
    """
    Wraps the Ollama LLM and exposes :meth:`classify` for single-email
    inference and :meth:`classify_batch` for sequential batch processing
    with inter-request throttling.

    Parameters
    ----------
    model:
        Ollama model tag (default: ``mistral:7b-instruct-q4_K_M``).
    base_url:
        Base URL of the Ollama server (default: ``http://localhost:11434``).
    inter_request_delay:
        Seconds to sleep between successive requests to prevent thermal
        throttling on constrained hardware (default: ``2.0``).
    num_thread:
        Number of CPU threads passed to Ollama (default: ``6``).
    timeout:
        HTTP request timeout in seconds (default: ``120.0``).
    max_retries:
        How many times to retry on transient errors before giving up
        (default: ``3``).
    retry_delay:
        Seconds to wait between retry attempts (default: ``5.0``).
    """

    def __init__(
        self,
        *,
        model:                str   = DEFAULT_MODEL,
        base_url:             str   = OLLAMA_BASE_URL,
        inter_request_delay:  float = 2.0,
        num_thread:           int   = 6,
        timeout:              float = 120.0,
        max_retries:          int   = 3,
        retry_delay:          float = 5.0,
    ) -> None:
        self.model                = model
        self.base_url             = base_url.rstrip("/")
        self.inter_request_delay  = inter_request_delay
        self.num_thread           = num_thread
        self.timeout              = timeout
        self.max_retries          = max_retries
        self.retry_delay          = retry_delay

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.ok
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Single-email inference
    # ------------------------------------------------------------------

    def classify(self, record: EmailRecord) -> ClassificationResult:
        """
        Run LLM inference on a single :class:`~preprocessing.EmailRecord`.

        Retries up to :attr:`max_retries` times on transient HTTP errors or
        JSON parse failures before re-raising.

        Parameters
        ----------
        record:
            A preprocessed email record from Stage 2.

        Returns
        -------
        ClassificationResult

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        user_prompt = _build_user_prompt(record)
        last_exc: Exception = RuntimeError("No attempts made.")

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = _call_ollama(
                    user_prompt,
                    model=self.model,
                    base_url=self.base_url,
                    num_thread=self.num_thread,
                    timeout=self.timeout,
                )
                result = _parse_classification(raw, record.message_id)
                logger.debug(
                    "Classified %s → category=%s action_required=%s",
                    record.message_id,
                    result.category,
                    result.action_required,
                )
                return result

            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d failed for message %s: %s",
                    attempt,
                    self.max_retries,
                    record.message_id,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"All {self.max_retries} attempts failed for message "
            f"{record.message_id}. Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        records: list[EmailRecord],
    ) -> list[ClassificationResult]:
        """
        Classify a list of emails sequentially, sleeping
        :attr:`inter_request_delay` seconds between each request.

        Failed messages are logged and skipped; the returned list may be
        shorter than *records* if any messages could not be classified.

        Parameters
        ----------
        records:
            List of preprocessed email records.

        Returns
        -------
        list[ClassificationResult]
            Results in the same order as *records*, omitting any failures.
        """
        results: list[ClassificationResult] = []

        for idx, record in enumerate(records):
            try:
                result = self.classify(record)
                results.append(result)
            except RuntimeError as exc:
                logger.error(
                    "Skipping message %s after exhausting retries: %s",
                    record.message_id,
                    exc,
                )

            # Throttle between requests to avoid thermal runaway — §4.3.
            if idx < len(records) - 1:
                time.sleep(self.inter_request_delay)

        logger.info(
            "Batch complete: %d/%d messages classified successfully.",
            len(results),
            len(records),
        )
        return results
