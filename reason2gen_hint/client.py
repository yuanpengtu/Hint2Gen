from __future__ import annotations

import os
import time
from typing import List, Optional

from openai import APITimeoutError, BadRequestError, OpenAI

from .rate_limit import RateLimiter


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str,
        limiter: RateLimiter,
        model: str,
        timeout: int = 180,
        max_retries: int = 8,
        base_url: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self.model = model
        self.limiter = limiter
        self.max_retries = max_retries

    def chat(self, messages: List[dict], max_tokens: int = 8192) -> str:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        maybe_params = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "response_format", "max_tokens"}
        for attempt in range(self.max_retries):
            try:
                self.limiter.acquire()
                resp = self.client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""
            except BadRequestError as exc:
                body = None
                try:
                    body = exc.response.json()
                except Exception:
                    body = None
                msg = (body or {}).get("error", {}).get("message", "")
                param = (body or {}).get("error", {}).get("param")
                if "context_length_exceeded" in msg:
                    raise
                if param in maybe_params and param in payload:
                    payload.pop(param, None)
                    continue
                raise
            except APITimeoutError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(10, 1.5 ** attempt))
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1.2 * (attempt + 1))
