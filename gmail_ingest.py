"""
quickstart.py
Gmail Local LLM Organizer — Stage 1 & 4: Ingestion + Output

Authenticates with the Gmail API, fetches messages incrementally, delegates
preprocessing and classification to the Stage 2/3 modules, then writes
results to a .jsonl file and SQLite database.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import os.path
import sqlite3

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from preprocessing import preprocess_gmail_message
from llm_interface import EmailClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
]

STATE_FILE           = "state.json"
OUTPUT_FILE          = "output.jsonl"
SQLITE_FILE          = "output.db"

BACKLOG_DAYS         = 30   # days to fetch on first run
MAX_MESSAGES_PER_RUN = 200  # safety cap per invocation

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_credentials() -> Credentials:
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds # type: ignore

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ---------------------------------------------------------------------------
# Gmail ingestion
# ---------------------------------------------------------------------------

def build_query(state: dict) -> str:
    """Return a Gmail search query based on stored state or default backlog."""
    if "last_processed_ts" in state:
        return f"after:{int(state['last_processed_ts'])}"
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=BACKLOG_DAYS)
    return f"after:{int(cutoff.timestamp())}"

def fetch_message_ids(service, query: str) -> list[str]:
    """Fetch message IDs matching *query*, paginating up to MAX_MESSAGES_PER_RUN."""
    ids: list[str] = []
    page_token = None
    while True:
        kwargs: dict = {"userId": "me", "q": query, "maxResults": 100}
        if page_token:
            kwargs["pageToken"] = page_token
        response = service.users().messages().list(**kwargs).execute()
        ids.extend(m["id"] for m in response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token or len(ids) >= MAX_MESSAGES_PER_RUN:
            break
    return ids[:MAX_MESSAGES_PER_RUN]

def fetch_full_message(service, msg_id: str) -> dict:
    """Fetch the full Gmail API message payload (format='full')."""
    return service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

# ---------------------------------------------------------------------------
# Output — .jsonl + SQLite
# ---------------------------------------------------------------------------

def init_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            message_id      TEXT PRIMARY KEY,
            subject         TEXT,
            sender          TEXT,
            date            TEXT,
            category        TEXT,
            action_required INTEGER,
            action_reason   TEXT,
            processed_at    TEXT
        )
    """)
    conn.commit()
    return conn

def write_record(record: dict, jsonl_file: str, db_conn: sqlite3.Connection) -> None:
    """Append *record* to .jsonl and upsert into SQLite."""
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    db_conn.execute("""
        INSERT OR REPLACE INTO emails
        (message_id, subject, sender, date, category, action_required, action_reason, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["message_id"],
        record["subject"],
        record["from"],
        record["date"],
        record["category"],
        int(record["action_required"]),
        record["action_reason"],
        record["processed_at"],
    ))
    db_conn.commit()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    creds   = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    state   = load_state()
    query   = build_query(state)

    logger.info("Fetching messages with query: %s", query)

    try:
        msg_ids = fetch_message_ids(service, query)
    except HttpError as e:
        logger.error("Error fetching message list: %s", e)
        return

    logger.info("Found %d messages to process.", len(msg_ids))

    classifier = EmailClassifier()
    if not classifier.ping():
        logger.error("Ollama server not reachable at %s. Is it running?", classifier.base_url)
        return

    db_conn   = init_sqlite(SQLITE_FILE)
    latest_ts = int(state.get("last_processed_ts", 0))
    processed_count = 0

    for idx, msg_id in enumerate(msg_ids):
        logger.info("Processing %s (%d/%d)...", msg_id, idx + 1, len(msg_ids))

        try:
            gmail_message = fetch_full_message(service, msg_id)

            # Stage 2: preprocess via preprocessing.py
            record = preprocess_gmail_message(gmail_message)

            # Stage 3: classify via llm_interface.py
            result = classifier.classify(record)

            internal_date_s = int(gmail_message.get("internalDate", 0)) // 1000

            output = {
                "message_id":      result.message_id,
                "subject":         record.subject,
                "from":            record.sender,
                "date":            record.date,
                "category":        result.category,
                "action_required": result.action_required,
                "action_reason":   result.action_reason,
                "processed_at":    datetime.datetime.utcnow().isoformat() + "Z",
            }

            write_record(output, OUTPUT_FILE, db_conn)

            if internal_date_s > latest_ts:
                latest_ts = internal_date_s

            processed_count += 1

        except HttpError as e:
            logger.error("Gmail API error for %s: %s", msg_id, e)
        except RuntimeError as e:
            logger.error("Classification failed for %s: %s", msg_id, e)
        except Exception as e:
            logger.exception("Unexpected error for %s: %s", msg_id, e)

    state["last_processed_ts"] = latest_ts
    save_state(state)
    db_conn.close()

    logger.info("Done. %d/%d messages processed.", processed_count, len(msg_ids))
    logger.info("Output written to %s and %s.", OUTPUT_FILE, SQLITE_FILE)


if __name__ == "__main__":
    main()
