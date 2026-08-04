import ipaddress
from multiprocessing import Value
import re
from tkinter import ALL
from urllib.parse import urlparse
from omega.storage.queue_queries import enqueue_ingestion_job

ALLOWED_INGESTION_SOURCE_TYPES = {"url", "text", "code"}
INGESTION_INTENT = re.compile(r"\b(save|add|ingest|import|store|keep)\b", re.IGNORECASE)


def is_explicit_ingestion_req(
    user_message: str,
    source_ref: str | None,
    raw_content: str | None,
) -> bool:
    if not INGESTION_INTENT.search(user_message):
        return False
    submitted_content = source_ref or raw_content
    return bool(
        isinstance(submitted_content, str)
        and submitted_content.strip()
        and submitted_content.strip() in user_message
    )

def validate_ingestion_req(
    source_type: str,
    source_ref: str | None,
    raw_content: str | None,
) -> None:
    if source_type not in ALLOWED_INGESTION_SOURCE_TYPES:
        raise ValueError("source type must be url, text, code")

    if source_type == "url":
        if raw_content is not None:
            raise ValueError("url ingestion cannot include raw content")
        if not source_ref:
            raise ValueError("url ingestion requires a URL")
        _validate_public_http_url(source_ref)
        return

    if not raw_content or not raw_content.strip():
        raise ValueError(f"{source_type} ingestion requires non-empty content")


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use http or https and include a host")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("local URLs cannot be ingested")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("private or local URLs cannnot be ingested")


async def enqueue_knowledge_ingestion(
    source_type: str,
    source_ref: str | None = None,
    raw_content: str | None = None,
    title: str | None = None
) -> dict:
    validate_ingestion_req(source_type, source_ref, raw_content)
    item_id, job_id, is_duplicate = await enqueue_ingestion_job(
        source_type=source_type,
        source_ref=source_ref,
        raw_content=raw_content,
        title=title
    )

    return {
        "item_id": str(item_id),
        "job_id": str(job_id) if job_id is not None else None,
        "duplicate": is_duplicate,
        "status": "already_exists" if is_duplicate else "queued",
    }