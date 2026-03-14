import base64
import hashlib
import hmac
import time

from fastapi import HTTPException, Request

from config import LOGIN_ENABLED, MAX_RPM, SESSION_COOKIE, SESSION_SECRET

_REQ_LOG: dict[str, list[float]] = {}


def rate_limit(request: Request):
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    q = _REQ_LOG.get(ip, [])
    q = [t for t in q if now - t < 60]
    if len(q) >= MAX_RPM:
        raise HTTPException(429, "Too many requests, slow down.")
    q.append(now)
    _REQ_LOG[ip] = q


def make_token(username: str) -> str:
    payload = (username or "user").encode("utf-8")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    data = base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
    return f"{data}.{sig}"


def verify_token(token: str) -> bool:
    try:
        data, sig = token.split(".", 1)
        payload = base64.urlsafe_b64decode(data + "===")
        expected = hmac.new(SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False


def is_authed(request: Request) -> bool:
    if not LOGIN_ENABLED:
        return True
    tok = request.cookies.get(SESSION_COOKIE)
    return bool(tok) and verify_token(tok)
