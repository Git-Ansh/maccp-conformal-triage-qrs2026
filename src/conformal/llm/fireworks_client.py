"""
Fireworks AI API client.

Calls DeepSeek V3 (or any model) via the OpenAI-compatible chat completions
endpoint. Includes MD5 caching for reproducibility and cost control.

Uses curl subprocess for HTTP calls (avoids Windows Python SSL issues in WSL).
Falls back to requests if curl is unavailable.
"""

import os
import json
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v3p2"

# Key file and cache directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
KEY_FILE = PROJECT_ROOT / ".fireworks_key"
CACHE_DIR = PROJECT_ROOT / "conformal_outputs" / "llm_cache"


def _load_api_key() -> Optional[str]:
    """Load API key from env var or .fireworks_key file."""
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return None


def _call_curl(url: str, headers: Dict, payload: Dict, timeout: int = 60) -> Dict:
    """Make HTTP POST via curl subprocess (bypasses Python SSL issues)."""
    # Write payload to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_file = f.name

    try:
        cmd = [
            "curl", "-s", "-X", "POST", url,
            "--max-time", str(timeout),
        ]
        for k, v in headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
        cmd.extend(["-d", f"@{payload_file}"])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 10
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr}")

        return json.loads(result.stdout)
    finally:
        try:
            os.unlink(payload_file)
        except OSError:
            pass


class FireworksClient:
    """
    HTTP client for Fireworks AI chat completions.

    Usage:
        client = FireworksClient()
        response = client.chat("What is 2+2?")

        # Consistency sampling (multiple runs with temperature)
        responses = client.chat_n(prompt, n=5, temperature=0.7)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = FIREWORKS_BASE_URL,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key or _load_api_key()
        if not self.api_key:
            raise ValueError(
                "FIREWORKS_API_KEY not set. Either:\n"
                "  export FIREWORKS_API_KEY='your-key'\n"
                f"  or write your key to {KEY_FILE}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Stats
        self.total_calls = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.total_tokens = 0

    def _cache_key(self, prompt: str, temperature: float, seed: Optional[int]) -> str:
        """Deterministic cache key from prompt + params."""
        key_str = f"{self.model}|{prompt}|{temperature}|{seed}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[str]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("response")
        return None

    def _save_cache(self, key: str, prompt: str, response: str, metadata: dict = None):
        path = self.cache_dir / f"{key}.json"
        data = {
            "model": self.model,
            "prompt": prompt[:500],  # Truncate for storage
            "response": response,
            "timestamp": time.time(),
        }
        if metadata:
            data.update(metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _api_call(self, messages: List[Dict], temperature: float,
                  max_tokens: int, seed: Optional[int]) -> str:
        """Make API call via curl, with retries."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        for attempt in range(self.max_retries):
            try:
                data = _call_curl(url, headers, payload)

                # Check for API errors
                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err))
                    if "rate" in msg.lower() or err.get("code") == 429:
                        wait = self.retry_delay * (2 ** attempt)
                        print(f"  Rate limited, waiting {wait:.0f}s...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"API error: {msg}")

                response_text = data["choices"][0]["message"]["content"]
                self.api_calls += 1
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                return response_text

            except (RuntimeError, KeyError, json.JSONDecodeError) as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"  API error: {e}. Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Fireworks API failed after {self.max_retries} retries: {e}"
                    )

        raise RuntimeError("No response from API")

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        seed: Optional[int] = 42,
        use_cache: bool = True,
    ) -> str:
        """
        Single chat completion call.

        Args:
            prompt: User message
            system: Optional system message
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Max response tokens
            seed: Random seed for reproducibility
            use_cache: Whether to use MD5 cache

        Returns:
            Response text string
        """
        self.total_calls += 1

        # Check cache
        if use_cache:
            cache_key = self._cache_key(prompt, temperature, seed)
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return cached

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response_text = self._api_call(messages, temperature, max_tokens, seed)

        # Save to cache
        if use_cache:
            self._save_cache(cache_key, prompt, response_text,
                             {"tokens": self.total_tokens})

        return response_text

    def chat_n(
        self,
        prompt: str,
        n: int = 5,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> List[str]:
        """
        Consistency sampling: call the LLM n times with temperature > 0.

        Used for confidence estimation -- if 4/5 responses agree, confidence = 0.8.

        Args:
            prompt: User message
            n: Number of samples
            system: Optional system message
            temperature: Must be > 0 for diversity
            max_tokens: Max response tokens

        Returns:
            List of n response strings
        """
        responses = []
        for i in range(n):
            resp = self.chat(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=i,  # Different seed each time
                use_cache=True,
            )
            responses.append(resp)
        return responses

    def stats(self) -> Dict:
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "cache_rate": self.cache_hits / max(1, self.total_calls),
        }
