from __future__ import annotations

from supabase import Client, create_client

from .config import get_settings


def get_supabase_client(service_role: bool = False) -> Client:
    settings = get_settings()
    key = (
        settings.supabase_service_role_key
        if service_role
        else settings.supabase_anon_key
    )
    return create_client(settings.supabase_url, key)
