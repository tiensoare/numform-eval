"""
Thin client over ARC's OpenAI-compatible LLM endpoints.

Two ARC options, both OpenAI-compatible (same code, different base_url + key):

  1) llm-api.arc.vt.edu  (persistent personal key)
       base_url = "https://llm-api.arc.vt.edu/api/v1"
       key      = from https://llm.arc.vt.edu  (Settings > Account > API keys)
       limits   = 10 concurrent requests, 8000 tokens / non-streaming request
       models   = gpt-oss-120b, GLM-5.2, Kimi-K3, DeepSeek-V4-Flash (+ thinking variants)

  2) OOD  (ood.arc.vt.edu, dedicated session)
       base_url = the session URL shown in the OOD app  (…/v1)
       key      = per-session key from the OOD app
       limits   = none (dedicated), but 1-hour idle timeout
       models   = 40+ from /common/data/models/

Access requires VT network or VPN in both cases.

Config is read from environment variables so keys never live in code:
  ARC_BASE_URL, ARC_API_KEY, ARC_MODEL
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # allow importing this module without openai installed
    OpenAI = None


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0          # deterministic for evaluation
    max_tokens: int = 2048
    reasoning_effort: Optional[str] = None  # e.g. "high" for gpt-oss / thinking models
    request_timeout: float = 120.0
    max_retries: int = 4

    @classmethod
    def from_env(cls) -> "LLMConfig":
        base = os.environ.get("ARC_BASE_URL", "https://llm-api.arc.vt.edu/api/v1")
        key = os.environ.get("ARC_API_KEY")
        # GLM-5.2 = ARC's strongest math/reasoning model; best default for
        # numerical tasks. Override with ARC_MODEL to compare others.
        model = os.environ.get("ARC_MODEL", "GLM-5.2")
        reasoning_effort = os.environ.get("REASONING_EFFORT") or None
        if not key:
            raise RuntimeError(
                "Set ARC_API_KEY (and optionally ARC_BASE_URL, ARC_MODEL). "
                "Get a key from https://llm.arc.vt.edu for llm-api, or from the "
                "OOD app for a dedicated session."
            )
        return cls(base_url=base, api_key=key, model=model,
                    reasoning_effort=reasoning_effort)


class LLMClient:
    """Wraps chat.completions with retries and simple token accounting."""

    def __init__(self, config: LLMConfig):
        if OpenAI is None:
            raise RuntimeError("pip install openai")
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def complete(self, prompt: str, system: str = "You are a careful math assistant.") -> dict:
        """
        Send one prompt. Returns dict with text, token counts, latency, and raw usage.
        Retries on transient errors with exponential backoff.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        last_err = None
        for attempt in range(self.config.max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    timeout=self.config.request_timeout, **kwargs
                )
                latency = time.time() - t0
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                if pt:
                    self.total_prompt_tokens += pt
                if ct:
                    self.total_completion_tokens += ct
                return {
                    "text": text,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "latency_s": latency,
                    "error": None,
                }
            except Exception as e:  # broad: network, rate, timeout, API errors
                last_err = e
                wait = 2 ** attempt
                time.sleep(wait)
        return {
            "text": "",
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_s": None,
            "error": str(last_err),
        }
