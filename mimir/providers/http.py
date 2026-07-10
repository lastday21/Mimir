from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import ProviderError


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ProviderError(f"{url} returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ProviderError(f"{url} is not reachable: {error.reason}") from error
    except TimeoutError as error:
        raise ProviderError(f"{url} timed out") from error
    except OSError as error:
        raise ProviderError(f"{url} failed: {error}") from error

    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProviderError(f"{url} returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise ProviderError(f"{url} returned unexpected JSON")
    return parsed
