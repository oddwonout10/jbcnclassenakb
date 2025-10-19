from __future__ import annotations

import hashlib
import re


def canonicalize_invitation_code(raw_code: str) -> str:
    """Normalize user-provided invitation code to the stored format."""
    if not raw_code:
        raise ValueError("Invitation code cannot be empty.")

    code = re.sub(r"[\s\-]", "", raw_code).upper()
    if len(code) != 8:
        raise ValueError("Invitation code must have 8 alphanumeric characters.")

    grouped = f"{code[:4]}-{code[4:]}"
    if not re.fullmatch(r"[A-Z0-9]{4}-[A-Z0-9]{4}", grouped):
        raise ValueError("Invitation code must contain only letters and numbers.")
    return grouped


def compute_invitation_hash(salt: str, code: str) -> str:
    payload = f"{salt}{code}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
