"""Configuration, model catalog, and persona presets for Sherra AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

APP_NAME = "Sherra AI"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_USAGE_URL = (
    "https://platform.openai.com/settings/organization/analytics/usage"
)

# Pricing per 1M tokens (USD) for rough session cost estimation in the playground.
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "label": "GPT-4o Mini (Fast & Affordable)",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "supports_temperature": True,
        "is_reasoning": False,
    },
    "gpt-4o": {
        "label": "GPT-4o (Flagship Multimodal)",
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "supports_temperature": True,
        "is_reasoning": False,
    },
    "gpt-4.1-mini": {
        "label": "GPT-4.1 Mini",
        "input_per_1m": 0.40,
        "output_per_1m": 1.60,
        "supports_temperature": True,
        "is_reasoning": False,
    },
    "gpt-4.1": {
        "label": "GPT-4.1",
        "input_per_1m": 2.00,
        "output_per_1m": 8.00,
        "supports_temperature": True,
        "is_reasoning": False,
    },
    "o4-mini": {
        "label": "o4-mini (Reasoning)",
        "input_per_1m": 1.10,
        "output_per_1m": 4.40,
        "supports_temperature": False,
        "is_reasoning": True,
    },
    "o3-mini": {
        "label": "o3-mini (Reasoning)",
        "input_per_1m": 1.10,
        "output_per_1m": 4.40,
        "supports_temperature": False,
        "is_reasoning": True,
    },
}

PERSONA_PRESETS: dict[str, str] = {
    "Sherra Default": (
        "You are Sherra AI, a helpful, accurate, and thoughtful AI assistant. "
        "Explain concepts clearly, use clean formatting, and be honest when you are unsure."
    ),
    "Python & Code Expert": (
        "You are Sherra AI, a senior Python software engineer. "
        "Write clean, well-typed, idiomatic Python code with concise explanations and edge-case awareness."
    ),
    "Concise Analyst": (
        "You are Sherra AI, a crisp analytical assistant. "
        "Give direct, structured answers using bullet points and short paragraphs without fluff."
    ),
    "Study Tutor": (
        "You are Sherra AI, a patient step-by-step tutor. "
        "Break complex topics into intuitive steps, use examples, and check understanding."
    ),
}


@dataclass
class PipelineConfig:
    """Runtime settings passed through every stage of the Sherra AI pipeline."""

    model: str = DEFAULT_MODEL
    system_prompt: str = PERSONA_PRESETS["Sherra Default"]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    max_history_messages: int = 20
    stream: bool = True
    api_key: str | None = None
    project_id: str | None = None


def load_env_settings() -> dict[str, str]:
    """Load OpenAI credentials and optional metadata from `.env` and environment."""
    load_dotenv(override=False)
    return {
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "project_id": os.getenv("OPENAI_PROJECT_ID", "").strip(),
        "usage_url": os.getenv("OPENAI_USAGE_URL", DEFAULT_USAGE_URL).strip()
        or DEFAULT_USAGE_URL,
    }


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a request based on MODEL_CATALOG pricing."""
    info = MODEL_CATALOG.get(model)
    if not info:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * info["input_per_1m"]
    output_cost = (completion_tokens / 1_000_000) * info["output_per_1m"]
    return round(input_cost + output_cost, 6)
