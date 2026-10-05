"""
middleware.py — Jurisdiction Access-Control & Enforcement Layer
===============================================================
Ensures that EVERY retrieval request is bound to a specific jurisdiction
derived from the authenticated user's session — never from caller input alone.

Architecture:
  • JurisdictionContext — immutable context attached to each request
  • JurisdictionEnforcer — resolves + validates jurisdiction from auth token
  • enforce_jurisdiction() — decorator / callable for use in FastAPI / Flask
  • MockAuthStore      — simple in-memory store for development/testing

In production, replace MockAuthStore.get_user_profile() with a call to your
real identity provider (JWT claims, OAuth scopes, database lookup, etc.).

Usage (FastAPI example):
    from middleware import JurisdictionEnforcer, JurisdictionContext

    enforcer = JurisdictionEnforcer()

    @app.post("/query")
    async def query(request: QueryRequest, token: str = Header(...)):
        ctx: JurisdictionContext = enforcer.from_token(token)
        result = retriever.answer(query=request.query, jurisdiction_ctx=ctx)
        return result
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from jurisdiction_registry import (
    JurisdictionConfig,
    get_jurisdiction_config,
    get_jurisdiction_key,
    normalize_jurisdiction,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Immutable per-request context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JurisdictionContext:
    """
    Immutable context bound to a single authenticated request.
    The retriever receives this object — it never accepts raw strings
    from callers for jurisdiction.
    """

    # Canonical jurisdiction name e.g. "Pakistan", "Hong Kong"
    canonical_name: str

    # Internal registry key e.g. "pakistan", "hong_kong"
    registry_key: str

    # Authenticated user id (for audit logging)
    user_id: str

    # Human-readable display name / email
    user_display: str = ""

    # Whether the user has elevated access (e.g. admin cross-jurisdiction)
    is_admin: bool = False

    # Additional allowed jurisdictions for admin users (empty = own only)
    extra_allowed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def config(self) -> Optional[JurisdictionConfig]:
        from jurisdiction_registry import JURISDICTION_REGISTRY
        return JURISDICTION_REGISTRY.get(self.registry_key)

    def allows(self, jurisdiction_name: str) -> bool:
        """True if this context permits access to the given jurisdiction."""
        if self.is_admin:
            return True
        norm = normalize_jurisdiction(jurisdiction_name)
        if norm.lower() == self.canonical_name.lower():
            return True
        # Extra allowances (multi-jurisdiction accounts)
        allowed_norms = {normalize_jurisdiction(j).lower() for j in self.extra_allowed}
        return norm.lower() in allowed_norms

    def assert_allows(self, jurisdiction_name: str) -> None:
        """Raise JurisdictionAccessError if access is not permitted."""
        if not self.allows(jurisdiction_name):
            raise JurisdictionAccessError(
                f"User '{self.user_id}' (jurisdiction: {self.canonical_name}) "
                f"attempted access to '{jurisdiction_name}'."
            )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class JurisdictionAccessError(PermissionError):
    """Raised when a user attempts to query outside their jurisdiction."""


class AuthenticationError(ValueError):
    """Raised when a token is invalid or expired."""


# ---------------------------------------------------------------------------
# User profile (returned by your identity provider)
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    user_id: str
    display_name: str
    jurisdiction: str        # canonical or alias
    is_admin: bool = False
    extra_jurisdictions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mock auth store — replace with DB / JWT validation in production
# ---------------------------------------------------------------------------

class MockAuthStore:
    """
    Development-only in-memory user store.

    In production replace with:
      - JWT decode + claims lookup
      - OAuth introspection
      - Database row fetch
    """

    _USERS: dict[str, UserProfile] = {
        # token_hash → UserProfile
        # Use sha256(token) as the key so raw tokens are never stored
        hashlib.sha256(b"tok-hk-user-001").hexdigest(): UserProfile(
            user_id="hk_001",
            display_name="Alice Chan",
            jurisdiction="Hong Kong",
        ),
        hashlib.sha256(b"tok-pak-user-001").hexdigest(): UserProfile(
            user_id="pak_001",
            display_name="Ali Hassan",
            jurisdiction="Pakistan",
        ),
        hashlib.sha256(b"tok-uk-user-001").hexdigest(): UserProfile(
            user_id="uk_001",
            display_name="James Smith",
            jurisdiction="United Kingdom",
        ),
        hashlib.sha256(b"tok-ca-user-001").hexdigest(): UserProfile(
            user_id="ca_001",
            display_name="Sarah Lee",
            jurisdiction="Canada",
        ),
        hashlib.sha256(b"tok-admin-001").hexdigest(): UserProfile(
            user_id="admin_001",
            display_name="System Admin",
            jurisdiction="Hong Kong",
            is_admin=True,
            extra_jurisdictions=["Pakistan", "United Kingdom", "Canada"],
        ),
    }

    @classmethod
    def get_user_profile(cls, token: str) -> Optional[UserProfile]:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return cls._USERS.get(token_hash)

    @classmethod
    def register_user(cls, token: str, profile: UserProfile) -> None:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cls._USERS[token_hash] = profile


# ---------------------------------------------------------------------------
# JWT-style token validator (stub — replace with real JWT lib)
# ---------------------------------------------------------------------------

class JWTValidator:
    """
    Minimal symmetric HMAC-based token validator.
    Replace with python-jose or PyJWT in production.

    Token format (base64-encoded JSON):
        {"sub": "user_id", "jurisdiction": "Pakistan",
         "is_admin": false, "exp": 1234567890}
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret.encode()

    def decode(self, token: str) -> dict:
        import base64
        try:
            parts = token.split(".")
            if len(parts) != 2:
                raise AuthenticationError("Malformed token")
            payload_b64, sig = parts
            expected_sig = hmac.new(
                self._secret,
                payload_b64.encode(),
                "sha256",
            ).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                raise AuthenticationError("Invalid token signature")
            payload = json.loads(base64.b64decode(payload_b64 + "==").decode())
            if payload.get("exp", 0) < time.time():
                raise AuthenticationError("Token expired")
            return payload
        except (ValueError, KeyError) as exc:
            raise AuthenticationError(f"Token decode failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Core enforcer
# ---------------------------------------------------------------------------

class JurisdictionEnforcer:
    """
    Resolves a JurisdictionContext from an auth token.

    Injection points:
      1. FastAPI dependency:  Depends(enforcer.from_token_dep)
      2. Flask before_request: ctx = enforcer.from_token(request.headers["X-Auth-Token"])
      3. CLI / batch:          ctx = enforcer.from_profile(profile)
    """

    def __init__(
        self,
        auth_store: Optional[MockAuthStore] = None,
        jwt_validator: Optional[JWTValidator] = None,
    ) -> None:
        self._store = auth_store or MockAuthStore()
        self._jwt = jwt_validator  # None → use MockAuthStore only

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def from_token(self, token: str) -> JurisdictionContext:
        """
        Resolve a JurisdictionContext from a bearer token.
        Tries JWT decode first (if validator configured), then mock store.
        Raises AuthenticationError if token is unrecognised.
        """
        if not token:
            raise AuthenticationError("No auth token provided")

        # Try JWT path
        if self._jwt:
            try:
                claims = self._jwt.decode(token)
                return self._context_from_claims(claims)
            except AuthenticationError:
                pass  # Fall through to mock store

        # Mock store path
        profile = self._store.get_user_profile(token)
        if not profile:
            raise AuthenticationError("Unrecognised token — access denied")
        return self._context_from_profile(profile)

    def from_profile(self, profile: UserProfile) -> JurisdictionContext:
        """Build a context directly from a UserProfile (for testing / CLI)."""
        return self._context_from_profile(profile)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _context_from_profile(self, profile: UserProfile) -> JurisdictionContext:
        canonical = normalize_jurisdiction(profile.jurisdiction)
        jkey = get_jurisdiction_key(canonical)
        if not jkey:
            raise AuthenticationError(
                f"Unknown jurisdiction in user profile: '{profile.jurisdiction}'"
            )
        log.info(
            "JurisdictionContext resolved: user=%s jurisdiction=%s admin=%s",
            profile.user_id,
            canonical,
            profile.is_admin,
        )
        return JurisdictionContext(
            canonical_name=canonical,
            registry_key=jkey,
            user_id=profile.user_id,
            user_display=profile.display_name,
            is_admin=profile.is_admin,
            extra_allowed=tuple(profile.extra_jurisdictions),
        )

    def _context_from_claims(self, claims: dict) -> JurisdictionContext:
        profile = UserProfile(
            user_id=str(claims.get("sub", "")),
            display_name=str(claims.get("name", "")),
            jurisdiction=str(claims.get("jurisdiction", "")),
            is_admin=bool(claims.get("is_admin", False)),
            extra_jurisdictions=list(claims.get("extra_jurisdictions", [])),
        )
        return self._context_from_profile(profile)

    # ------------------------------------------------------------------
    # FastAPI dependency helper
    # ------------------------------------------------------------------

    async def from_token_dep(self, authorization: str = "") -> JurisdictionContext:
        """
        FastAPI-compatible async dependency.

        Example:
            from fastapi import Header, Depends
            from middleware import JurisdictionEnforcer

            enforcer = JurisdictionEnforcer()

            @app.post("/query")
            async def query_endpoint(
                request: QueryRequest,
                ctx: JurisdictionContext = Depends(enforcer.from_token_dep),
                authorization: str = Header(default=""),
            ):
                ...
        """
        token = authorization.removeprefix("Bearer ").strip()
        return self.from_token(token)


# ---------------------------------------------------------------------------
# Convenience: build a no-auth development context
# ---------------------------------------------------------------------------

def dev_context(
    jurisdiction: str,
    user_id: str = "dev_user",
) -> JurisdictionContext:
    """
    Build a JurisdictionContext without auth, for local development / testing.
    Do NOT use in production.
    """
    canonical = normalize_jurisdiction(jurisdiction)
    jkey = get_jurisdiction_key(canonical)
    if not jkey:
        raise ValueError(f"Unknown jurisdiction: '{jurisdiction}'")
    return JurisdictionContext(
        canonical_name=canonical,
        registry_key=jkey,
        user_id=user_id,
        user_display="Dev User",
        is_admin=False,
    )