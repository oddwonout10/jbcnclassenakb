from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Iterable

from .config import get_settings


def send_email(
    subject: str,
    body_text: str,
    *,
    to: Iterable[str] | None = None,
    body_html: str | None = None,
) -> None:
    """Send an email using the configured SMTP credentials."""
    settings = get_settings()

    recipients = list(to) if to else [settings.escalation_email_to]
    if not recipients:
        raise ValueError("No recipients specified for email.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.escalation_email_from
    message["To"] = ", ".join(recipients)
    message.set_content(body_text)
    if body_html:
        message.add_alternative(body_html, subtype="html")

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(settings.smtp_username, settings.smtp_password)
        smtp.send_message(message)
