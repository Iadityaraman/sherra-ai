"""Pure Python JSON conversation persistence for Sherra AI."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sherra_ai.config import DEFAULT_MODEL, PERSONA_PRESETS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatSession:
    """Serializable chat session mimicking ChatGPT conversation history."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New Chat"
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    model: str = DEFAULT_MODEL
    system_prompt: str = PERSONA_PRESETS["Sherra Default"]
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_message(
        self,
        role: str,
        content: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": _utc_now_iso(),
        }
        if trace is not None:
            entry["trace"] = trace
        self.messages.append(entry)
        if self.title == "New Chat" and role == "user":
            clean = " ".join(content.split())
            self.title = (clean[:42] + "...") if len(clean) > 42 else (clean or "New Chat")
        self.updated_at = _utc_now_iso()


class ConversationStore:
    """Stores Sherra AI conversations as JSON files in a local directory."""

    def __init__(self, directory: str | Path = "conversations") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
        if not safe_id:
            raise ValueError("Invalid session ID.")
        return self.directory / f"{safe_id}.json"

    def save(self, session: ChatSession) -> None:
        session.updated_at = _utc_now_iso()
        path = self._path_for(session.id)
        path.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8")

    def load(self, session_id: str) -> ChatSession | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChatSession(**data)

    def list_sessions(self) -> list[ChatSession]:
        sessions: list[ChatSession] = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(ChatSession(**data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def export_markdown(session: ChatSession) -> str:
        lines = [
            f"# {session.title}",
            f"- **Assistant:** Sherra AI",
            f"- **Model:** `{session.model}`",
            f"- **Total Tokens:** {session.total_tokens}",
            f"- **Est. Cost:** ${session.total_cost_usd:.6f}",
            "",
            "---",
            "",
        ]
        for msg in session.messages:
            speaker = "User" if msg["role"] == "user" else "Sherra AI"
            lines.append(f"### {speaker}\n")
            lines.append(msg["content"])
            lines.append("")
        return "\n".join(lines)
