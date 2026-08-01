import logging
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from omega.storage.memory_queries import (
    get_unconsolidated_count,
    mark_entries_consolidated,
    apply_importance_decay,
    count_total_msgs
)
from omega.storage.retrieval_queries import search_memory_entries
from omega.storage.postgres_session import db_pool
from omega.embeddings.embedding_service import get_embedding_service
from omega.llm.client import get_llm_provider
from omega.memory.core import read_memory_md, read_user_md
from omega.environment.conf_loader import omega_settings
from omega.memory.profile_infer import append_section_to_profile, safe_auto_write

logger = logging.getLogger("Consolidation")

CONSOLIDATION_TRIGGER_COUNT = 10
MERGE_SIMILARITY_THRESHOLD = 0.15
DECAY_DAYS_THRESHOLD = 7
DECAY_FACTOR = 0.95
DECAY_FLOOR = 0.05
CORE_MEMORY_MAX_ENTRIES = 10

PATTERN_PROMPT = """You are Omega's memory curator. Review these recent memory entries
(session summaries and extracted facts) and identify TWO things:

1. RECURRING PATTERNS: Any themes, topics, or behaviors that appear across multiple entries.
   Simple, concrete observations only. Examples: "mentioned work stress in 4 of the last 6 sessions",
   "frequently references the same project", "prefers concise answers".
   If nothing stands out, say "No clear patterns yet."

2. CORE FACTS: Which entries represent the most important, durable information about the user -
   things the agent should always know without searching. These are the entries that should go into
   the agent's always-loaded core memory file (MEMORY.md). Pick at most 5-10.

Return ONLY a JSON object with this exact structure (no explanations, no markdown):
{
    "patterns": ["pattern 1", "pattern 2"],
    "core_entry_ids": ["uuid1", "uuid2", "uuid3"]
}"""

PROFILE_PROMPT = """You are omega's profile curator. Review the recent session below and identify any new information about the USER that should be added to USER.md
Your notes about who the user is and how to interact with them.

USER.md should contain:
- Identity: name, role, background - anything the user has revealed about themselves
- Communication style: how the user talks, their tone, verbosity preference
- Preferences: how they like you to respond, what they value
- Work context: what they're working on, their ambitions
- Life: how they've been recently, what their day looks like
- Constraints: anything you should avoid or be aware of

CURRENT USER.md:
{current_user_md}

RECENT SESSION:
{recent_session}

Based on these sessions, what NEW information (not already in USER.md) should be added?
Focus on durable observations - communication patterns seen across multiple sessions,
identify details the user has shared, or explicit preferences they've stated.

If nothing new should be added, return an empty updates list.

Return ONLY a JSON object with this structure:
{
    "updates": [
        {
            "section_title": "Communication Style",
            "content": "The user consistently sends very short, direct messages. They prefer brief answers and seem to get annoyed by long explanations. Keep responses to 1-3 sentences unless asked for detail"
        }
    ],
    "observations": "Brief note about overall communication patterns observed"
}

Rules:
- Each update should be a natural-language note to yourself, not structured data
- Only propose updates when you see clear, consistent patterns.
- Don't repeat information already in USER.md
- Write as notes to yourself: "The user prefers...", "User mentioned...", "User works with..."
- If nothing clear, return empty updates list"""

class ConsolidationJob:
    def __init__(self):
        self.llm_client = get_llm_provider()
        self.embedding_service = get_embedding_service()

    async def trigger_if_needed(self) -> bool:
        count = await get_unconsolidated_count()
        if count < CONSOLIDATION_TRIGGER_COUNT:
            logger.debug(f"consolidated not triggered: {count} < {CONSOLIDATION_TRIGGER_COUNT}")
            return False

        logger.info(f"consolidation triggered: {count} unconsolidated entries >= {CONSOLIDATION_TRIGGER_COUNT}")
        try:
            await self.run()
            logger.info("consolidated completed successfully")
            return True
        except Exception as e:
            logger.error(f"consolidation failed: {e}")
            return False

    async def run(self):
        entries = await self._fetch_unconsolidated()
        
        if not entries:
            logger.info("no unconsolidated entries to process")
            return

        entry_ids = [e["id"] for e in entries]
        await self._decay_stale_entries(entries)
        merge_map = await self._find_merges(entries)
        await self._apply_merges(merge_map)
        pattern_result = await self._identify_patterns_and_core(entries, merge_map)
        core_ids = pattern_result.get("core_entry_ids", [])
        await self._write_memory_md(entries, core_ids, merge_map)
        await self._regenerate_user_md()
        await mark_entries_consolidated(entry_ids)
        patterns = pattern_result.get("patterns", [])
        if patterns and patterns != ["No clear patterns yet."]:
            logger.info(f"patterns noted: {patterns}")

    async def _fetch_unconsolidated(self) -> list[dict]:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, memory_type, content, embedding::text, importance,
                       access_count, last_accessed_at, created_at, metadata,
                       source_session_id
                FROM memory_entries
                WHERE (metadata->>'consolidated' IS NULL
                       OR metadata->>'consolidated' = 'false')
                  AND superseded_by IS NULL
                ORDER BY created_at DESC 
            """)
        return [dict(r) for r in rows]

    async def _decay_stale_entries(self, entries: list[dict]):
        stale_cutoff = datetime.now(timezone.utc) - timedelta(days=DECAY_DAYS_THRESHOLD)
        stale_ids = []

        for entry in entries:
            last_access = entry.get("last_accessed_at")
            importance = entry.get("importance", 0.5)

            if importance <= DECAY_FLOOR:
                continue

            if last_access is None:
                created = entry.get("created_at")
                if created and created.replace(tzinfo=timezone.utc) < stale_cutoff:
                    stale_ids.append(entry["id"])
            elif isinstance(last_access, datetime):
                if last_access.replace(tzinfo=timezone.utc) if last_access.tzinfo is None else last_access < stale_cutoff:
                    stale_ids.append(entry["id"])

        if stale_ids:
            logger.info(f"decaying importance on {len(stale_ids)} stale entries")
            await apply_importance_decay(stale_ids, DECAY_FACTOR)
            
    async def _find_merges(self, entries: list[dict]) -> dict[str, set[str]]:
        merge_map = {}
        already_merged = set()

        for i, e1 in enumerate(entries):
            if e1["id"] in already_merged:
                continue

            emb1_str = e1.get("embedding")
            if not emb1_str:
                continue
            try:
                emb1 = json.loads(emb1_str) if isinstance(emb1_str, str) else emb1_str
            except (json.JSONDecodeError, TypeError):
                continue

            for j in range(i + 1, len(entries)):
                e2 = entries[j]
                if e2["id"] in already_merged:
                    continue
                
                emb2_str = e2.get("embedding")
                if not emb2_str:
                    continue
                try:
                    emb2 = json.loads(emb2_str) if isinstance(emb2_str, str) else emb2_str
                except (json.JSONDecodeError, TypeError):
                    continue

                sim = self._cosine_similarity(emb1, emb2)
                if sim > (1.0 - MERGE_SIMILARITY_THRESHOLD):
                    sc1 = (e1.get("importance", 0.5) or 0.5) * (1 + __import__("math").log((e1.get("access_count", 0) or 0) + 1))
                    sc2 = (e2.get("importance", 0.5) or 0.5) * (1 + __import__("math").log((e2.get("access_count", 0) or 0) + 1))

                    if sc1 >= sc2:
                        keeper, merged = e1["id"], e2["id"]
                    else:
                        keeper, merged = e2["id"], e1["id"]

                    if keeper not in merge_map:
                        merge_map[keeper] = set()
                    merge_map[keeper].add(merged)
                    already_merged.add(merged)

        if merge_map:
            total_merged = sum(len(v) for v in merge_map.values())
            logger.info(f"found {len(merge_map)} merge groups ({total_merged} entries to merge)")
        
        return merge_map

    async def _apply_merges(self, merge_map: dict[str, set[str]]):
        if not merge_map:
            return

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                for keeper_id, merged_ids in merge_map.items():
                    for merged_id in merged_ids:
                        await conn.execute(
                            "UPDATE memory_entries SET superseded_by = $1 WHERE id = $2",
                            keeper_id, merged_id
                        )
                        logger.info(f"merge: entry {merged_id} superseded by {keeper_id}")

        total = sum(len(v) for v in merge_map.values())
        logger.info(f"applied {total} merge supersedings across {len(merge_map)} groups")

    async def _identify_patterns_and_core(
        self, entries: list[dict], merge_map: dict[str, set[str]]
    ) -> dict:
        keeper_ids = set(merge_map.keys())
        merged_ids = set()
        for mids in merge_map.values():
            merged_ids.update(mids)

        active_entries = [e for e in entries if e["id"] not in merged_ids]

        if not active_entries:
            return {"patterns": [], "core_entry_ids": []}

        lines = []
        for i, entry in enumerate(active_entries[:50]):
            keep_note = ""
            if entry["id"] in keeper_ids:
                kept_count = len(merge_map[entry["id"]])
                keep_note = f" [ABSORBED {kept_count} similar entries]"
            lines.append(
                f"Entry {i+1} (id={entry['id']}, type={entry['memory_type']}, "
                f"importance={entry.get('importance', 0.5):.2f}, "
                f"access_count={entry.get('access_count', 0)}){keep_note}\n"
                f" {entry['content']}\n"
            )

        entries_text = "\n".join(lines)
        user_prompt = f"MEMORY ENTRIES:\n\n{entries_text}"

        try:
            response = await self.llm_client.generate_json(PATTERN_PROMPT, user_prompt)
            import json
            result = json.loads(response)
            logger.info(f"LLM consolidation: {len(result.get('patterns', []))} patterns, "
                        f"{len(result.get('core_entry_ids', []))} core entries")
            return result
        except Exception as e:
            logger.error(f"LLM consolidation failed: {e}")
            return {"patterns": ["LLM pattern extraction failed"], "core_entry_ids": []}

    async def _write_memory_md(
        self, entries: list[dict], core_ids: list[str], merge_map: dict[str, set[str]]
    ):
        id_to_entry = {e["id"]: e for e in entries}

        scored = []
        for eid in core_ids:
            entry = id_to_entry.get(eid)
            if not entry:
                continue
            score = (
                (entry.get("importance", 0.5) or 0.5) * 1.0
                + math.log((entry.get("access_count", 0) or 0) + 1) * 0.5
            )
            scored.append((score, entry))

        for keeper_id, merged_ids in merge_map.items():
            if keeper_id not in core_ids:
                entry = id_to_entry.get(keeper_id)
                if entry:
                    score = 2.0 + len(merged_ids) * 0.5
                    scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        lines = ["# Omega Memory", "",
                  "Curated by the consolidation job. Edit this file directly to correct anything",
                  f"Last updated: {__import__('datetime').datetime.now().isoformat()}", ""]

        count = 0
        for _, entry in scored:
            if count >= CORE_MEMORY_MAX_ENTRIES:
                break
            memory_type = entry.get("memory_type", "unknown")
            label = "FACT" if memory_type == "extracted_fact" else "SESSION"
            lines.append(f"## [{label}] {entry['content'].strip()}")
            lines.append("")
            count += 1

        if count == 0:
            lines.append("_No core memories yet_")

        content = "\n".join(lines)
        memory_dir = Path(omega_settings.memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_path = memory_dir / "MEMORY.md"

        old_content = ""
        if memory_path.exists():
            old_content = memory_path.read_text()

        was_written, reason = safe_auto_write("MEMORY", content)
        if was_written:
            logger.info(f"Wrote MEMORY.md: {count} entries ({len(content)} bytes)")
            if old_content != content:
                logger.info("MEMORY.md changed - core memory updated")
        else:
            logger.warning(f"MEMORY.md write skipped: {reason}")

    async def _reflect_profile(self):
        try:
            total_messages = await count_total_msgs()
            min_messages = omega_settings.profile_inference_min_messages
            if total_messages < min_messages:
                logger.info(
                    f"profile reflection skipped: warm-up - {total_messages} total messages"
                    f"(need {min_messages})"
                )
                return

            recent_msgs = await self._fetch_msgs_for_reflection()
            if not recent_msgs:
                logger.info("profile reflection: no recent messages available")
                return

            current_user_md = read_user_md()
            if not current_user_md:
                current_user_md = "(USER.md is empty - no profile yet)"

            messages_text = self._format_msgs_refl(recent_msgs)
            prompt = PROFILE_PROMPT.format(
                current_user_md=current_user_md,
                messages_text=messages_text,
            )

            response = await self.llm_client.generate_json(prompt, "")
            result = json.loads(response)
            updates = result.get("updates", [])
            observations = result.get("observations", "")

            if observations:
                logger.info(f"profile reflection observations: {observations}")

            if not updates:
                logger.info("profile reflection: no updates proposed")
                return

            await self._apply_profile_updates(updates)

        except json.JSONDecodeError as e:
            logger.error(f"profile reflecton: LLM returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"profile reflection failed (non_fatal): {e}")

    async def _fetch_msgs_for_reflection(self) -> list[dict]:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content, created_at, session_id
                FROM messages
                WHERE role IN ('user', 'assistant')
                ORDER BY created_at DESC
                LIMIT 20
            """)
        messages = [dict(r) for r in rows]
        messages.reverse()
        return messages

    def _format_msgs_refl(self, messages: list[dict]) -> str:
        MAX_MSG_LEN = 500
        lines = []
        current_session = None

        for msg in messages:
            sid = str(msg.get("session_id", ""))[:8]
            role = msg["role"].upper()
            content = msg.get("content", "")

            if len(content) > MAX_MSG_LEN:
                content = content[:MAX_MSG_LEN] + " ..."

            if sid != current_session:
                current_session = sid
                lines.append(f"\n--- Session {sid} ---")

            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    async def _apply_profile_updates(self, updates: list[dict]):

        for update in updates:
            title = update.get("section_title", "Update").strip()
            content = update.get("content", "").strip()
            if not content:
                continue

            current = read_user_md()
            if title in current:
                logger.debug(f"profile reflection: section '{title}' already in USER.md, skipping")
                continue

            was_written, reason = append_section_to_profile("USER", content, title)
            if was_written:
                logger.info(f"profile reflection: added section '{title}' to USER.md")
            else:
                logger.debug(f"profile reflection: failed to add '{title}' - {reason}")

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

_consolidation_job: ConsolidationJob | None = None

def get_consolidation_job() -> ConsolidationJob:
    global _consolidation_job
    if _consolidation_job is None:
        _consolidation_job = ConsolidationJob()
    return _consolidation_job