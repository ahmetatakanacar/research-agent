import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import HTTPException, Request

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    "content-security-policy": CONTENT_SECURITY_POLICY,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware, so streaming (SSE) responses are not buffered or wrapped."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        private = scope["path"].startswith("/api/")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                extra = dict(SECURITY_HEADERS)
                if private:
                    extra["x-robots-tag"] = "noindex"
                    extra["cache-control"] = "no-store"
                headers += [(k.encode(), v.encode()) for k, v in extra.items() if k.encode() not in present]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RateLimiter:
    def __init__(self, limit: int, window: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1.0, self.window - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > 5000:
                for stale in [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]:
                    del self._hits[stale]
            return None


def client_key(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def supplied_api_key(request: Request) -> str:
    header = request.headers.get("x-api-key", "").strip()
    if header:
        return header
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def check_api_key(request: Request, expected: str) -> None:
    if not expected:
        return
    supplied = supplied_api_key(request)
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.", headers={"WWW-Authenticate": "Bearer"})
