import logging
import hashlib
import json
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger("ProfileFiles")

def _hash_path(file_stem: str) -> Path:
    memory_dir = Path(omega_settings.memory_dir)
    return memory_dir / f".{file_stem}.last_auto_hash"

def _file_path(file_stem: str) -> Path:
    memory_dir = Path(omega_settings.memory_dir)
    return memory_dir / f"{file_stem}.md"

def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def was_man_edited(file_stem: str) -> tuple[bool, str]:
    path = _file_path(file_stem)
    hash_path = _hash_path(file_stem)

    if not path.exists():
        return False, f"{file_stem}.md does not exist yet"

    current_hash = _compute_hash(path.read_text(encoding="utf-8"))

    if not hash_path.exists():
        hash_path.write_text(current_hash)
        return False, f"first run, stored baseline hash for {file_stem}.md"

    stored_hash = hash_path.read_text().strip()

    if current_hash != stored_hash:
        return True, (
            f"{file_stem}.md was manually edited - hash mismatch "
            f"(stored={stored_hash[:12]}..., current={current_hash[:12]}...)"
        )

    return False, f"{file_stem}.md unchanged since last auto-write"

def _record_auto_write(file_stem: str, content: str):
    hash_path = _hash_path(file_stem)
    hash_path.write_text(_compute_hash(content))

def safe_auto_write(file_stem: str, content: str, force: bool = False) -> tuple[bool, str]:
    if not force:
        edited, reason = was_man_edited(file_stem)
        if edited:
            logger.warning(
                f"skipping auto-write to {file_stem}.md: {reason}"
                "preserving manual user edits, remove the last auto hash file"
            )
            return False, reason

    memory_dir = Path(omega_settings.memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(memory_dir), suffix=".tmp")
    try:
        os.write(fd, content.encode())
        os.fsync(fd)
    finally:
        os.close(fd)

    os.rename(tmp_path, str(_file_path(file_stem)))
    _record_auto_write(file_stem, content)

    logger.info(f"wrote {file_stem}.md ({len(content)} chars)")
    return True, f"wrote {file_stem}.md ({len(content)} chars)"

def _record_profile_provenance(
    file_stem: str,
    text: str,
    section_title: str,
    provenance: dict,
):
    path = Path(omega_settings.memory_dir) / f".{file_stem}.provenance.jsonl"
    record = {
        **provenance,
        "target_file": f"{file_stem}.md",
        "section_title": section_title,
        "content": text,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

def append_section_to_profile(file_stem: str, text: str, section_title: str = "", force:bool = False, provenance: dict | None = None,) -> tuple[bool, str]:
    from omega.memory.core import read_memory_md, read_user_md

    if provenance is None and not force:
        return False, "direct user provenance is required for profile writes"

    if not force:
        edited, reason = was_man_edited(file_stem)
        if edited:
            logger.warning(
                f"skipping append to {file_stem}.md: user manually edited"
                "Append would risk clobbering manual changes"
            )
            return False, f"skipped: {file_stem}.md was manually edited"

    if file_stem == "MEMORY":
        current = read_memory_md()
    else:
        current = read_user_md()

    new_section = ""
    if section_title:
        new_section += f"\n\n## {section_title}\n"
    new_section += f"{text}\n"
    new_content = current.rstrip() + "\n" + new_section
    was_written, reason = safe_auto_write(file_stem, new_content, force=force)
    if was_written and provenance is not None:
        _record_profile_provenance(file_stem, text, section_title, provenance)
    return was_written, reason