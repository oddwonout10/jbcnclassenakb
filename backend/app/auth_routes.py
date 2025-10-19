from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import re

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, constr, field_validator
from supabase import Client

from .supabase_client import get_supabase_client
from .utils import canonicalize_invitation_code, compute_invitation_hash

router = APIRouter(prefix="/auth", tags=["auth"])


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupRequest(BaseModel):
    email: constr(min_length=5, max_length=255)  # type: ignore[valid-type]
    password: constr(min_length=8)  # type: ignore[valid-type]
    invitation_code: str = Field(..., min_length=1)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        email = value.strip()
        if not EMAIL_REGEX.match(email):
            raise ValueError("Enter a valid email address.")
        return email.lower()


class SignupResponse(BaseModel):
    guardian_id: str
    student_id: str
    student_name: str
    message: str


def _find_student_by_code(client: Client, code: str) -> dict[str, Any] | None:
    response = (
        client.table("students")
        .select(
            "id, full_name, grade, section, invitation_code_salt, "
            "invitation_code_hash, invitation_code_revealed, mother_name, "
            "mother_phone, father_name, father_phone"
        )
        .execute()
    )
    for student in response.data or []:
        expected_hash = compute_invitation_hash(student["invitation_code_salt"], code)
        if expected_hash == student["invitation_code_hash"]:
            return student
    return None


def _guardian_exists(client: Client, student_id: str) -> bool:
    response = (
        client.table("guardians").select("id").eq("student_id", student_id).execute()
    )
    return bool(response.data)


def _create_supabase_user(client: Client, email: str, password: str) -> Any:
    try:
        result = client.auth.admin.create_user(
            {
                "email": email,
                "password": password,
                "email_confirm": True,
            }
        )
    except Exception as exc:  # pragma: no cover - surfacing auth errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to create user: {exc}",
        ) from exc

    user = getattr(result, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation returned no user object.",
        )
    return user


@router.post("/signup", response_model=SignupResponse)
def invitation_signup(payload: SignupRequest) -> SignupResponse:
    try:
        canonical_code = canonicalize_invitation_code(payload.invitation_code)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    client = get_supabase_client(service_role=True)
    student = _find_student_by_code(client, canonical_code)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation code not recognized. Please contact the class parent.",
        )

    if _guardian_exists(client, student_id=student["id"]):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A guardian account already exists for this student. Contact the class parent to reset.",
        )

    user = _create_supabase_user(client, payload.email, payload.password)

    now_iso = datetime.now(timezone.utc).isoformat()
    guardian_record = {
        "student_id": student["id"],
        "auth_user_id": getattr(user, "id"),
        "email": payload.email.lower(),
        "mother_name": student.get("mother_name"),
        "mother_phone": student.get("mother_phone"),
        "father_name": student.get("father_name"),
        "father_phone": student.get("father_phone"),
        "registered_at": now_iso,
        "last_login_at": now_iso,
    }

    insert_response = client.table("guardians").insert(guardian_record).execute()
    if not insert_response.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store guardian details.",
        )

    client.table("students").update(
        {
            "invitation_code_revealed": True,
            "updated_at": now_iso,
        }
    ).eq("id", student["id"]).execute()

    guardian_id = insert_response.data[0]["id"]
    return SignupResponse(
        guardian_id=guardian_id,
        student_id=student["id"],
        student_name=student["full_name"],
        message="Signup successful. You can now log in with your email and password.",
    )
