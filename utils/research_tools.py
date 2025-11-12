import os
import time
import json
import requests
from collections import OrderedDict
from typing import Generator, Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    """
    Minimal Groq chat client with retries, timeouts, and optional streaming.

    Notes on common 400 errors and fixes:
    - invalid_request_error: Ensure a valid model (e.g., "llama3-70b-8192") and set max_tokens > 0.
    - authentication_error: Ensure GROQ_API_KEY is set and has the correct prefix (gsk_...).
    - message format: messages must be a list of {role, content}.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = GROQ_URL):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Add it to your .env or environment variables."
            )
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        self.timeout = 30
        self.max_retries = 3
        self.backoff = 1.5
        self._min_interval = 0.2  # basic rate limit to avoid spamming
        self._last_call = 0.0
        # tiny in-memory LRU cache to accelerate repeated prompts
        self._cache = OrderedDict()
        self._cache_limit = 64

    def _sleep_for_rate_limit(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _post(self, payload: dict, stream: bool = False):
        self._sleep_for_rate_limit()
        attempt = 0
        while True:
            try:
                resp = self.session.post(
                    self.base_url,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                    stream=stream,
                )
                if resp.status_code >= 400 and not stream:
                    # Try to surface API error details
                    try:
                        err = resp.json()
                    except Exception:
                        err = {"error": {"message": resp.text}}
                    raise requests.HTTPError(
                        f"{resp.status_code} {resp.reason}: {err}", response=resp
                    )
                return resp
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                attempt += 1
                # Retry for transient cases
                status = getattr(getattr(e, "response", None), "status_code", None)
                transient = status in {429, 500, 502, 503, 504} or isinstance(
                    e, (requests.Timeout, requests.ConnectionError)
                )
                if attempt <= self.max_retries and transient:
                    time.sleep(self.backoff ** attempt)
                    continue
                raise

    def chat(
        self,
        prompt: str,
        *,
    model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        system: Optional[str] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(1, int(max_tokens)),
            "top_p": top_p,
        }
        # cache key
        key = json.dumps({
            "model": model,
            "system": system or "",
            "prompt": prompt,
            "temperature": round(float(temperature), 2),
            "max_tokens": int(max_tokens),
            "top_p": round(float(top_p), 2),
        }, sort_keys=True)
        if key in self._cache:
            # move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        try:
            resp = self._post(payload)
            data = resp.json()
            out = data["choices"][0]["message"]["content"].strip()
            # store in cache
            self._cache[key] = out
            if len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
            return out
        except Exception as e:
            return self._format_error(e)

    def chat_stream(
        self,
        prompt: str,
        *,
    model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(1, int(max_tokens)),
            "top_p": top_p,
            "stream": True,
        }

        try:
            resp = self._post(payload, stream=True)
            accumulated = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            accumulated += delta
                            yield accumulated
                    except Exception:
                        # if parsing fails, ignore this line
                        continue
        except Exception as e:
            yield self._format_error(e)

    @staticmethod
    def _format_error(e: Exception) -> str:
        status = getattr(getattr(e, "response", None), "status_code", None)
        reason = getattr(getattr(e, "response", None), "reason", "")
        try:
            body = getattr(e, "response").json() if getattr(e, "response", None) else None
        except Exception:
            body = getattr(getattr(e, "response", None), "text", None)
        hint = ""
        if status == 400:
            hint = (
                " Check the model name and ensure max_tokens > 0. Also verify the messages payload."
            )
        elif status == 401:
            hint = " Check your GROQ_API_KEY."
        elif status in {429, 500, 502, 503, 504}:
            hint = " This may be transient; it was retried automatically."
        return (
            f"Error communicating with Groq API: {status or ''} {reason or ''} {str(e)}\n{body or ''}{hint}"
        )


# Backwards-compatible function used by the app
_client: Optional[GroqClient] = None


def _get_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient()
    return _client


def query_groq(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 1.0,
    system: Optional[str] = None,
):
    client = _get_client()
    return client.chat(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        system=system,
    )
