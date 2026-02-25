"""Email client — async Gmail send/receive via App Password.

Uses aiosmtplib for sending and aioimaplib for IMAP inbox access.
Credentials come from config (env vars or .accounts file).
"""

from __future__ import annotations

import email
import json
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import aiosmtplib
from aioimaplib import IMAP4_SSL

from .config import config

logger = logging.getLogger("meridian.email")


class EmailClient:
    """Async Gmail client using App Password auth."""

    def __init__(self):
        self.address = config.gmail_address
        self.password = config.gmail_app_password
        self._imap: IMAP4_SSL | None = None

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
    ) -> dict:
        """Send an email via Gmail SMTP."""
        if not self.address or not self.password:
            return {"sent": False, "error": "Gmail credentials not configured"}

        msg = MIMEMultipart()
        msg["From"] = f"GigaClaude <{self.address}>"
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc

        msg.attach(MIMEText(body, "plain"))

        recipients = [to]
        if cc:
            recipients.extend(addr.strip() for addr in cc.split(","))

        try:
            await aiosmtplib.send(
                msg,
                hostname="smtp.gmail.com",
                port=587,
                start_tls=True,
                username=self.address,
                password=self.password,
                recipients=recipients,
            )
            logger.info(f"Email sent to {to}: {subject}")
            return {"sent": True, "to": to, "subject": subject}
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return {"sent": False, "error": str(e)}

    async def _get_imap(self) -> IMAP4_SSL:
        """Get or create IMAP connection."""
        if self._imap is None:
            self._imap = IMAP4_SSL(host="imap.gmail.com")
            await self._imap.wait_hello_from_server()
            await self._imap.login(self.address, self.password)
        return self._imap

    async def fetch_inbox(
        self,
        limit: int = 10,
        unread_only: bool = True,
    ) -> list[dict]:
        """Fetch recent emails from inbox."""
        if not self.address or not self.password:
            return []

        try:
            imap = await self._get_imap()
            await imap.select("INBOX")

            search_criteria = "UNSEEN" if unread_only else "ALL"
            _, data = await imap.search(search_criteria)

            if not data or not data[0]:
                return []

            # Get UIDs — most recent first
            uids = data[0].split()
            uids = uids[-limit:]  # last N
            uids.reverse()

            messages = []
            for uid in uids:
                _, msg_data = await imap.fetch(uid.decode(), "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] UID)")
                if msg_data and len(msg_data) >= 2:
                    raw = msg_data[1]
                    if isinstance(raw, bytes):
                        parsed = email.message_from_bytes(raw)
                    else:
                        parsed = email.message_from_string(str(raw))

                    messages.append({
                        "uid": uid.decode(),
                        "from": parsed.get("From", "unknown"),
                        "subject": parsed.get("Subject", "(no subject)"),
                        "date": parsed.get("Date", ""),
                    })

            return messages

        except Exception as e:
            logger.error(f"Inbox fetch failed: {e}")
            self._imap = None  # Reset connection on error
            return []

    async def read_email(self, uid: str) -> dict:
        """Read full email body by UID."""
        if not self.address or not self.password:
            return {"error": "Gmail credentials not configured"}

        try:
            imap = await self._get_imap()
            await imap.select("INBOX")

            _, msg_data = await imap.fetch(uid, "(RFC822)")
            if not msg_data or len(msg_data) < 2:
                return {"error": f"Message {uid} not found"}

            raw = msg_data[1]
            if isinstance(raw, bytes):
                parsed = email.message_from_bytes(raw)
            else:
                parsed = email.message_from_string(str(raw))

            # Extract body
            body = ""
            if parsed.is_multipart():
                for part in parsed.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="replace")
                            break
            else:
                payload = parsed.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")

            return {
                "uid": uid,
                "from": parsed.get("From", "unknown"),
                "to": parsed.get("To", ""),
                "subject": parsed.get("Subject", "(no subject)"),
                "date": parsed.get("Date", ""),
                "body": body[:5000],  # Cap body length
            }

        except Exception as e:
            logger.error(f"Email read failed: {e}")
            self._imap = None
            return {"error": str(e)}

    async def close(self):
        """Close IMAP connection."""
        if self._imap:
            try:
                await self._imap.logout()
            except Exception:
                pass
            self._imap = None
