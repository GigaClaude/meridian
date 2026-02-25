"""Email client — Gmail API with OAuth2.

Uses google-api-python-client for send/read via Gmail API.
OAuth2 token comes from a token file (generated via one-time auth flow).
Auto-refreshes expired tokens.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from .config import config

logger = logging.getLogger("meridian.email")

TOKEN_PATH = config.gmail_token_path


def _get_creds() -> Credentials | None:
    """Load and refresh OAuth2 credentials."""
    if not TOKEN_PATH.exists():
        logger.warning(f"Gmail token not found at {TOKEN_PATH}")
        return None

    try:
        data = json.loads(TOKEN_PATH.read_text())
        creds = Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes"),
        )

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save refreshed token
            data["token"] = creds.token
            TOKEN_PATH.write_text(json.dumps(data, indent=2))
            logger.info("Gmail token refreshed")

        return creds
    except Exception as e:
        logger.error(f"Failed to load Gmail credentials: {e}")
        return None


def _get_service():
    """Build Gmail API service."""
    creds = _get_creds()
    if not creds:
        return None
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


class EmailClient:
    """Gmail API client with OAuth2."""

    def __init__(self):
        self.address = config.gmail_address
        self._service = None

    def _get_service(self):
        if self._service is None:
            self._service = _get_service()
        return self._service

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
    ) -> dict:
        """Send an email via Gmail API."""
        service = self._get_service()
        if not service:
            return {"sent": False, "error": "Gmail not configured (missing OAuth token)"}

        msg = MIMEMultipart()
        msg["From"] = f"GigaClaude <{self.address}>"
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        msg.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

        try:
            result = await asyncio.to_thread(
                service.users().messages().send(
                    userId="me", body={"raw": raw}
                ).execute
            )
            logger.info(f"Email sent to {to}: {subject}")
            return {"sent": True, "to": to, "subject": subject, "id": result.get("id")}
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            self._service = None  # Reset on error
            return {"sent": False, "error": str(e)}

    async def fetch_inbox(
        self,
        limit: int = 10,
        unread_only: bool = True,
    ) -> list[dict]:
        """Fetch recent emails from inbox."""
        service = self._get_service()
        if not service:
            return []

        try:
            query = "in:inbox"
            if unread_only:
                query += " is:unread"

            result = await asyncio.to_thread(
                service.users().messages().list(
                    userId="me", q=query, maxResults=limit
                ).execute
            )

            messages = result.get("messages", [])
            if not messages:
                return []

            inbox = []
            for msg_ref in messages:
                msg = await asyncio.to_thread(
                    service.users().messages().get(
                        userId="me", id=msg_ref["id"], format="metadata",
                        metadataHeaders=["From", "Subject", "Date"],
                    ).execute
                )

                headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                inbox.append({
                    "uid": msg_ref["id"],
                    "from": headers.get("From", "unknown"),
                    "subject": headers.get("Subject", "(no subject)"),
                    "date": headers.get("Date", ""),
                })

            return inbox

        except Exception as e:
            logger.error(f"Inbox fetch failed: {e}")
            self._service = None
            return []

    async def read_email(self, uid: str) -> dict:
        """Read full email by message ID."""
        service = self._get_service()
        if not service:
            return {"error": "Gmail not configured"}

        try:
            msg = await asyncio.to_thread(
                service.users().messages().get(
                    userId="me", id=uid, format="full",
                ).execute
            )

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

            # Extract body
            body = ""
            payload = msg.get("payload", {})

            if "parts" in payload:
                for part in payload["parts"]:
                    if part.get("mimeType") == "text/plain":
                        data = part.get("body", {}).get("data", "")
                        if data:
                            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                            break
            elif payload.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(
                    payload["body"]["data"]
                ).decode("utf-8", errors="replace")

            return {
                "uid": uid,
                "from": headers.get("From", "unknown"),
                "to": headers.get("To", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "body": body[:5000],
            }

        except Exception as e:
            logger.error(f"Email read failed: {e}")
            self._service = None
            return {"error": str(e)}

    async def close(self):
        """Cleanup."""
        self._service = None
