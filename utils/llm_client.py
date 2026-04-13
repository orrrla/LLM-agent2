from __future__ import annotations

import json
from typing import Any

import requests

from config.runtime import get_app_settings
from utils import logger


def call_text_model(
    model: str,
    user_prompt: str,
    *,
    max_tokens: int = 1200,
    timeout: float = 20.0,
    temperature: float = 0.2,
) -> str:
    if not model:
        return ""
    if _is_gemini_model(model):
        return _call_gemini(
            model=model,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
        )
    return _call_openai_compatible(
        model=model,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )


def _call_openai_compatible(
    *,
    model: str,
    user_prompt: str,
    max_tokens: int,
    timeout: float,
    temperature: float,
) -> str:
    settings = get_app_settings()
    headers = {
        "Authorization": settings.api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            settings.base_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"] or "")
    except Exception as exc:
        logger.warning(f"openai-compatible llm failed: {exc}")
        return ""


def _call_gemini(
    *,
    model: str,
    user_prompt: str,
    max_tokens: int,
    timeout: float,
    temperature: float,
) -> str:
    settings = get_app_settings()
    if not settings.gemini_api_key:
        logger.warning("gemini api key is missing")
        return ""

    model_name = model.replace("models/", "").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.gemini_api_key}"
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [str(part.get("text", "")) for part in parts if part.get("text")]
        return "".join(text_parts).strip()
    except Exception as exc:
        logger.warning(f"gemini llm failed: {exc}")
        return ""


def _is_gemini_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gemini") or normalized.startswith("models/gemini")
