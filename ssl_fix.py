"""
Windows SSL workaround: skip corrupted Windows cert-store entries.

aiohttp (pulled in by google-genai) calls ssl.create_default_context() at
import time. On some Windows machines that raises:
  ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data

This module replaces create_default_context so it loads CA certs from
certifi instead of the Windows certificate store.
"""

from __future__ import annotations

import ssl

_patched = False


def apply() -> None:
    global _patched
    if _patched:
        return

    try:
        import certifi
    except ImportError:
        return

    _original = ssl.create_default_context

    def _create_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: str | bytes | None = None,
    ) -> ssl.SSLContext:
        if cafile or capath or cadata:
            return _original(purpose, cafile=cafile, capath=capath, cadata=cadata)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=certifi.where())
        return ctx

    ssl.create_default_context = _create_default_context  # type: ignore[assignment]
    _patched = True


apply()
