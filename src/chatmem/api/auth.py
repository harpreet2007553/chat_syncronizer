"""Bearer token authentication and secure key management for chatmem API."""
from __future__ import annotations

from pathlib import Path
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

security_bearer = HTTPBearer(auto_error=False)


def get_or_create_auth_token(token_path_str: str) -> str:
    """Load or generate a cryptographically secure random bearer token.

    The token is stored outside version control. We log and display the path,
    never the secret token content itself.
    """
    path = Path(token_path_str).expanduser().resolve()
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            logger.info("Loaded auth token from %s", path)
            return token

    # Generate a fresh 256-bit secret token
    token = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")

    try:
        path.chmod(0o600)
    except Exception:
        pass

    logger.info("Generated new auth token and stored at %s", path)
    print(f"[chatmem] Generated new auth token at: {path}")
    return token


class TokenVerifier:
    """Dependency that validates bearer tokens using constant-time comparison."""

    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    def __call__(
        self,
        credentials: HTTPAuthorizationCredentials | None = Security(security_bearer),
    ) -> str:
        if not credentials or not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not secrets.compare_digest(credentials.credentials, self.expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials

