"""MongoDB connection management for Backline.

Provides a lazily-initialised MongoClient and convenience accessors for the
default database and its collections.  The connection string is read from
the ``MONGODB_URI`` environment variable.

When ``MONGODB_URI`` is not set the module exposes ``None`` values so that
callers can fall back to local CSV loading without crashing on import.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

_client = None
_db = None


def _resolve_dns_records(name: str, record_type: str):
    """Resolve DNS records with a public-resolver fallback."""
    import dns.resolver

    resolvers = [dns.resolver.Resolver()]

    public_resolver = dns.resolver.Resolver(configure=False)
    public_resolver.nameservers = ["1.1.1.1", "8.8.8.8"]
    public_resolver.timeout = 5
    public_resolver.lifetime = 10
    resolvers.append(public_resolver)

    last_error = None
    for resolver in resolvers:
        try:
            return list(resolver.resolve(name, record_type))
        except Exception as exc:  # pragma: no cover - network-path specific
            last_error = exc

    if last_error is not None:
        raise last_error
    return []


def _normalize_srv_uri(uri: str) -> str:
    """Convert mongodb+srv URIs into seed-list URIs with DNS fallback."""
    if not uri.startswith("mongodb+srv://"):
        return uri

    parsed = urlsplit(uri)
    if "@" in parsed.netloc:
        credentials, cluster_host = parsed.netloc.rsplit("@", 1)
        auth_prefix = f"{credentials}@"
    else:
        cluster_host = parsed.netloc
        auth_prefix = ""

    srv_records = _resolve_dns_records(f"_mongodb._tcp.{cluster_host}", "SRV")
    txt_records = _resolve_dns_records(cluster_host, "TXT")

    hosts = ",".join(
        sorted(f"{record.target.to_text(omit_final_dot=True)}:{record.port}" for record in srv_records)
    )

    txt_options: dict[str, str] = {}
    for record in txt_records:
        txt_value = "".join(
            part.decode("utf-8") if isinstance(part, bytes) else part for part in record.strings
        )
        txt_options.update(dict(parse_qsl(txt_value, keep_blank_values=True)))

    query_options = dict(parse_qsl(parsed.query, keep_blank_values=True))
    combined_options = dict(txt_options)
    combined_options.update(query_options)
    combined_options.setdefault("tls", "true")

    return urlunsplit(
        (
            "mongodb",
            f"{auth_prefix}{hosts}",
            parsed.path,
            urlencode(combined_options),
            parsed.fragment,
        )
    )


def mongo_client_kwargs() -> dict[str, Any]:
    """Return the TLS defaults used for Atlas connections."""
    import certifi

    return {"tlsCAFile": certifi.where()}


def create_client(uri: str | None = None, **kwargs):
    """Create a Mongo client with the project's TLS defaults."""
    if uri is None:
        uri = os.environ.get("MONGODB_URI")
    if not uri:
        return None
    uri = _normalize_srv_uri(uri)

    from pymongo import MongoClient

    client_kwargs = mongo_client_kwargs()
    client_kwargs.update(kwargs)
    return MongoClient(uri, **client_kwargs)


def format_connection_error(exc: Exception, *, uri: str | None = None) -> str:
    """Return a short, actionable MongoDB connection error message."""
    message = str(exc)
    normalized = message.lower()
    hints: list[str] = []

    if "all nameservers failed" in normalized or "servfail" in normalized:
        hints.append("The local DNS resolver failed to resolve Atlas SRV/TXT records.")
        hints.append("Retry on a different network or switch your DNS resolver to 1.1.1.1 or 8.8.8.8.")

    if "ssl handshake failed" in normalized or "tlsv1 alert internal error" in normalized:
        hints.append("Atlas rejected the TLS handshake before authentication completed.")
        if uri and "mongodb.net" in uri:
            hints.append(
                "Check Atlas Network Access and add this machine's current public IP to the IP access list."
            )
        hints.append(
            "If you are on a VPN, corporate proxy, or antivirus with TLS inspection, allow outbound TLS traffic to Atlas on port 27017."
        )
    elif "serverselectiontimeouterror" in normalized or "replicasetnoprimary" in normalized:
        hints.append("MongoDB Atlas did not present a reachable primary during server selection.")

    if not hints:
        return message

    return "\n".join([message, "", "Likely fixes:"] + [f"- {hint}" for hint in hints])


def _get_client():
    global _client
    if _client is None:
        _client = create_client()
        if _client is None:
            return None
    return _client


def get_db() -> "Database | None":
    """Return the default Backline database, or ``None`` when unconfigured."""
    global _db
    if _db is None:
        client = _get_client()
        if client is None:
            return None
        _db = client.get_default_database("backline")
    return _db


def get_collection(name: str) -> "Collection | None":
    """Return a collection handle, or ``None`` when MongoDB is unavailable."""
    db = get_db()
    if db is None:
        return None
    return db[name]
