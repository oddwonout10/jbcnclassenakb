from __future__ import annotations

import json
from functools import lru_cache
from typing import Tuple

import requests
from openai import OpenAI
from requests import HTTPError

from .config import Settings


class LLMClientError(RuntimeError):
    """Raised when the configured LLM provider cannot fulfil a request."""


@lru_cache(maxsize=1)
def _get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _call_openai(prompt: str, settings: Settings) -> Tuple[str, str]:
    if not settings.openai_api_key:
        raise LLMClientError("OPENAI_API_KEY is not configured.")
    client = _get_openai_client(settings.openai_api_key)
    response = client.responses.create(
        model=settings.openai_model,
        input=prompt,
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=1024,
    )
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip(), settings.openai_model

    parts = []
    for item in getattr(response, "output", []) or []:
        if item.get("type") == "output_text" and item.get("text"):
            parts.append(item["text"])
    if not parts:
        raise LLMClientError("OpenAI response did not contain any text output.")
    return "\n".join(parts).strip(), settings.openai_model


def _call_gemini(prompt: str, settings: Settings) -> Tuple[str, str]:
    if not settings.gemini_api_key:
        raise LLMClientError("GEMINI_API_KEY is not configured.")
    model = settings.gemini_model or "gemini-pro"
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.9,
            "maxOutputTokens": 1024,
        },
    }
    try:
        response = requests.post(
            endpoint,
            params={"key": settings.gemini_api_key},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except HTTPError as exc:
        raise LLMClientError(f"Gemini request failed: {exc}") from exc

    data = response.json()
    if "error" in data:
        raise LLMClientError(
            f"Gemini returned an error: {json.dumps(data['error'])}"
        )

    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        texts = [part.get("text") for part in parts if part.get("text")]
        if texts:
            return "\n".join(texts).strip(), model
    raise LLMClientError("Gemini response did not contain any text output.")


def _call_groq(prompt: str, settings: Settings) -> Tuple[str, str]:
    if not settings.groq_api_key:
        raise LLMClientError("GROQ_API_KEY is not configured.")
    model = settings.groq_model or "llama3-70b-8192"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except HTTPError as exc:
        raise LLMClientError(f"Groq request failed: {exc}") from exc

    data = response.json()
    if "error" in data:
        raise LLMClientError(
            f"Groq returned an error: {json.dumps(data['error'])}"
        )

    choices = data.get("choices") or []
    if not choices:
        raise LLMClientError("Groq response did not contain any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise LLMClientError("Groq response did not contain message content.")
    return content.strip(), model


def _call_anthropic(prompt: str, settings: Settings) -> Tuple[str, str]:
    if not settings.anthropic_api_key:
        raise LLMClientError("ANTHROPIC_API_KEY is not configured.")
    model = settings.anthropic_model or "claude-3-haiku-20240307"
    endpoint = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }
    try:
        response = requests.post(
            endpoint,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except HTTPError as exc:
        raise LLMClientError(f"Anthropic request failed: {exc}") from exc

    data = response.json()
    if "error" in data:
        raise LLMClientError(
            f"Anthropic returned an error: {json.dumps(data['error'])}"
        )

    blocks = data.get("content") or []
    texts = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                texts.append(block["text"].strip())
        elif hasattr(block, "text") and block.text:
            texts.append(block.text.strip())
    if not texts:
        raise LLMClientError("Anthropic response did not contain text content.")
    return "\n".join(texts), model


def _call_cohere(prompt: str, settings: Settings) -> Tuple[str, str]:
    if not settings.cohere_api_key:
        raise LLMClientError("COHERE_API_KEY is not configured.")
    model = settings.cohere_model or "command"
    endpoint = "https://api.cohere.com/v1/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.3,
        "p": 0.9,
    }
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.cohere_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except HTTPError as exc:
        raise LLMClientError(f"Cohere request failed: {exc}") from exc

    data = response.json()
    if "error" in data:
        raise LLMClientError(
            f"Cohere returned an error: {json.dumps(data['error'])}"
        )

    generations = data.get("generations") or []
    if not generations:
        raise LLMClientError("Cohere response did not contain generations.")
    first = generations[0]
    text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
    if not text:
        raise LLMClientError("Cohere response did not contain text content.")
    return text.strip(), model


def generate_answer(prompt: str, settings: Settings) -> Tuple[str, str]:
    provider = (settings.llm_provider or "openai").lower()
    if provider == "openai":
        return _call_openai(prompt, settings)
    if provider in {"gemini", "google"}:
        return _call_gemini(prompt, settings)
    if provider == "groq":
        return _call_groq(prompt, settings)
    if provider == "anthropic":
        return _call_anthropic(prompt, settings)
    if provider == "cohere":
        return _call_cohere(prompt, settings)
    raise LLMClientError(
        f"Unsupported LLM provider '{settings.llm_provider}'. "
        "Supported providers: openai, gemini, groq, anthropic, cohere."
    )
