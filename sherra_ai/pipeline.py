"""Modular Python LLM pipeline for Sherra AI."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from sherra_ai.config import MODEL_CATALOG, PipelineConfig, estimate_cost_usd


@dataclass
class StageLog:
    """Execution record for a single pipeline stage."""

    stage_name: str
    duration_ms: float
    status: str
    detail: str = ""


@dataclass
class UsageStats:
    """Token, timing, and cost metrics for one pipeline run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0


@dataclass
class PipelineContext:
    """Mutable context object passed sequentially through pipeline stages."""

    user_input: str
    history: list[dict[str, str]] = field(default_factory=list)
    config: PipelineConfig = field(default_factory=PipelineConfig)
    normalized_input: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    response_text: str = ""
    usage: UsageStats = field(default_factory=UsageStats)
    stage_logs: list[StageLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_log(
        self, stage_name: str, duration_ms: float, status: str = "ok", detail: str = ""
    ) -> None:
        self.stage_logs.append(
            StageLog(
                stage_name=stage_name,
                duration_ms=round(duration_ms, 2),
                status=status,
                detail=detail,
            )
        )


class PipelineStage(ABC):
    """Base class for synchronous stages in the Sherra AI pipeline."""

    name: str = "stage"

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Transform or enrich `ctx` and return it."""


class InputPreprocessorStage(PipelineStage):
    """Sanitize and validate user input before building the prompt."""

    name = "1. Input Preprocessor"

    def __init__(self, max_chars: int = 32_000) -> None:
        self.max_chars = max_chars

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        cleaned = ctx.user_input.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty.")
        if len(cleaned) > self.max_chars:
            raise ValueError(
                f"Prompt exceeds maximum length of {self.max_chars} characters."
            )

        ctx.normalized_input = cleaned
        ctx.metadata["input_chars"] = len(cleaned)
        ctx.metadata["input_words"] = len(cleaned.split())
        ctx.add_log(
            self.name,
            (time.perf_counter() - t0) * 1000,
            detail=f"{ctx.metadata['input_words']} words, {ctx.metadata['input_chars']} chars",
        )
        return ctx


class PromptBuilderStage(PipelineStage):
    """Assemble the system prompt, trimmed history window, and current user turn."""

    name = "2. Context & Prompt Builder"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        messages: list[dict[str, str]] = []

        system_prompt = (ctx.config.system_prompt or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        limit = max(0, ctx.config.max_history_messages)
        recent_history = ctx.history[-limit:] if limit > 0 else []
        valid_roles = {"user", "assistant", "system"}
        trimmed_count = 0
        for item in recent_history:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in valid_roles and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
                trimmed_count += 1

        messages.append({"role": "user", "content": ctx.normalized_input})
        ctx.messages = messages
        ctx.metadata["history_messages_used"] = trimmed_count
        ctx.metadata["total_payload_messages"] = len(messages)

        ctx.add_log(
            self.name,
            (time.perf_counter() - t0) * 1000,
            detail=f"Built {len(messages)} messages ({trimmed_count} history)",
        )
        return ctx


class OpenAIInferenceStage:
    """Call OpenAI Chat Completions API (supports both streaming and single-shot)."""

    name = "3. OpenAI LLM Inference"

    def __init__(self, client_factory: Callable[[PipelineConfig], OpenAI] | None = None) -> None:
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(config: PipelineConfig) -> OpenAI:
        api_key = (config.api_key or "").strip()
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_API_KEY in .env or enter it in the sidebar."
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        project_id = (config.project_id or "").strip()
        # Project-scoped keys (sk-proj-...) already encode project routing,
        # while passing project explicitly is safe if it matches the key.
        if project_id and not api_key.startswith("sk-proj-"):
            kwargs["project"] = project_id
        return OpenAI(**kwargs)

    @staticmethod
    def _build_request_kwargs(ctx: PipelineContext, stream: bool) -> dict[str, Any]:
        model_info = MODEL_CATALOG.get(ctx.config.model, {})
        is_reasoning = bool(model_info.get("is_reasoning", False)) or ctx.config.model.startswith(("o1", "o3", "o4"))
        supports_temperature = bool(model_info.get("supports_temperature", not is_reasoning))

        kwargs: dict[str, Any] = {
            "model": ctx.config.model,
            "messages": ctx.messages,
            "stream": stream,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = ctx.config.max_tokens
        else:
            kwargs["max_tokens"] = ctx.config.max_tokens

        if supports_temperature:
            kwargs["temperature"] = ctx.config.temperature
            kwargs["top_p"] = ctx.config.top_p

        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        client = self._client_factory(ctx.config)
        kwargs = self._build_request_kwargs(ctx, stream=False)
        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        ctx.response_text = choice.message.content or ""
        if getattr(response, "usage", None):
            ctx.usage.prompt_tokens = response.usage.prompt_tokens or 0
            ctx.usage.completion_tokens = response.usage.completion_tokens or 0
            ctx.usage.total_tokens = response.usage.total_tokens or (
                ctx.usage.prompt_tokens + ctx.usage.completion_tokens
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ctx.usage.latency_ms = round(elapsed_ms, 2)
        ctx.add_log(
            self.name,
            elapsed_ms,
            detail=f"model={ctx.config.model}, tokens={ctx.usage.total_tokens}",
        )
        return ctx

    def stream_process(self, ctx: PipelineContext) -> Iterator[str]:
        """Yield response text deltas and record usage + timing when finished."""
        t0 = time.perf_counter()
        client = self._client_factory(ctx.config)
        kwargs = self._build_request_kwargs(ctx, stream=True)
        stream_resp = client.chat.completions.create(**kwargs)

        chunks: list[str] = []
        for event in stream_resp:
            if getattr(event, "usage", None):
                ctx.usage.prompt_tokens = event.usage.prompt_tokens or 0
                ctx.usage.completion_tokens = event.usage.completion_tokens or 0
                ctx.usage.total_tokens = event.usage.total_tokens or (
                    ctx.usage.prompt_tokens + ctx.usage.completion_tokens
                )
            if event.choices:
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    chunks.append(text)
                    yield text

        ctx.response_text = "".join(chunks)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ctx.usage.latency_ms = round(elapsed_ms, 2)
        ctx.add_log(
            self.name,
            elapsed_ms,
            detail=f"model={ctx.config.model} (streamed), tokens={ctx.usage.total_tokens}",
        )


class ResponsePostprocessorStage(PipelineStage):
    """Finalize output formatting and compute estimated request cost."""

    name = "4. Response Postprocessor"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        ctx.response_text = ctx.response_text.strip()
        ctx.usage.estimated_cost_usd = estimate_cost_usd(
            ctx.config.model,
            ctx.usage.prompt_tokens,
            ctx.usage.completion_tokens,
        )
        ctx.add_log(
            self.name,
            (time.perf_counter() - t0) * 1000,
            detail=(
                f"{len(ctx.response_text)} chars, "
                f"est. ${ctx.usage.estimated_cost_usd:.6f}"
            ),
        )
        return ctx


class SherraPipeline:
    """Orchestrates pre-processing, prompt assembly, OpenAI inference, and post-processing."""

    def __init__(
        self,
        pre_stages: list[PipelineStage] | None = None,
        inference_stage: OpenAIInferenceStage | None = None,
        post_stages: list[PipelineStage] | None = None,
    ) -> None:
        self.pre_stages = (
            pre_stages
            if pre_stages is not None
            else [InputPreprocessorStage(), PromptBuilderStage()]
        )
        self.inference_stage = inference_stage or OpenAIInferenceStage()
        self.post_stages = (
            post_stages
            if post_stages is not None
            else [ResponsePostprocessorStage()]
        )

    def stage_names(self) -> list[str]:
        """Return ordered human-readable names of all stages in the pipeline."""
        return (
            [stage.name for stage in self.pre_stages]
            + [self.inference_stage.name]
            + [stage.name for stage in self.post_stages]
        )

    def run(
        self,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        config: PipelineConfig | None = None,
    ) -> PipelineContext:
        """Run the full pipeline synchronously and return the populated context."""
        ctx = PipelineContext(
            user_input=user_input,
            history=list(history or []),
            config=config or PipelineConfig(),
        )
        for stage in self.pre_stages:
            ctx = stage.process(ctx)
        ctx = self.inference_stage.process(ctx)
        for stage in self.post_stages:
            ctx = stage.process(ctx)
        return ctx

    def run_stream(
        self,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        config: PipelineConfig | None = None,
    ) -> tuple[PipelineContext, Iterator[str]]:
        """Execute pre-stages immediately and return `(ctx, token_iterator)`."""
        ctx = PipelineContext(
            user_input=user_input,
            history=list(history or []),
            config=config or PipelineConfig(),
        )
        for stage in self.pre_stages:
            ctx = stage.process(ctx)

        def _stream_and_finalize() -> Iterator[str]:
            yield from self.inference_stage.stream_process(ctx)
            for stage in self.post_stages:
                stage.process(ctx)

        return ctx, _stream_and_finalize()
