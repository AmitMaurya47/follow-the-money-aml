"""Dummy authentication for the prototype UI.

One demo account. The password is stored only as a salted PBKDF2 hash. This is NOT real
security: there is no user store, no rate limiting, and no session token. Replace it
before exposing the app to anyone.
"""
from __future__ import annotations

import hashlib
import hmac

DEMO_USER = "demo"
DEMO_PASSWORD_HINT = "demo123"  # shown on the sign-in page; it is a demo login
_SALT = b"follow-the-money-demo-salt"
_ITERATIONS = 100_000
_STORED_HASH = hashlib.pbkdf2_hmac("sha256", DEMO_PASSWORD_HINT.encode(), _SALT, _ITERATIONS).hex()


def check_login(username: str, password: str) -> bool:
    if not username or not password:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), _SALT, _ITERATIONS).hex()
    user_ok = hmac.compare_digest(username.strip().lower(), DEMO_USER)
    pass_ok = hmac.compare_digest(digest, _STORED_HASH)
    return user_ok and pass_ok
