from __future__ import annotations

from textwrap import dedent
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .email import send_email

router = APIRouter(prefix="/escalations", tags=["escalations"])


class EscalationTestRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    guardian_name: Optional[str] = Field(None, max_length=200)
    guardian_email: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)


class EscalationTestResponse(BaseModel):
    status: str
    message: str


@router.post("/test", response_model=EscalationTestResponse)
def trigger_test_escalation(
    payload: EscalationTestRequest,
    settings: Settings = Depends(get_settings),
) -> EscalationTestResponse:
    """Utility endpoint to verify SMTP configuration."""
    try:
        body = dedent(
            f"""
            A test escalation email was requested from the knowledge base backend.

            Guardian name : {payload.guardian_name or 'N/A'}
            Guardian email: {payload.guardian_email or 'N/A'}

            Question:
            {payload.question}

            Notes:
            {payload.notes or 'None'}
            """
        ).strip()

        send_email(
            subject="[Test] Class QA escalation check",
            body_text=body,
            to=[settings.escalation_email_to],
        )
    except Exception as exc:  # pragma: no cover - capture SMTP issues
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to send escalation email: {exc}",
        ) from exc

    return EscalationTestResponse(
        status="sent",
        message=f"Email sent to {settings.escalation_email_to}",
    )
