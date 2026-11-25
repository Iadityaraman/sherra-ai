"""Sherra AI — Pure Python ChatGPT-style Bot & LLM Playground (Streamlit UI)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import streamlit as st
from openai import APIError, AuthenticationError, RateLimitError

from sherra_ai import (
    APP_NAME,
    DEFAULT_MODEL,
    MODEL_CATALOG,
    PERSONA_PRESETS,
    ChatSession,
    ConversationStore,
    PipelineConfig,
    PipelineContext,
    SherraPipeline,
    load_env_settings,
)

STARTER_PROMPTS = [
    "Explain how transformer attention works in simple terms.",
    "Write a clean Python function to parse and validate JSON logs.",
    "Help me plan a 7-day study roadmap for Linear Algebra.",
    "Compare REST vs GraphQL with concrete trade-offs.",
]


def _init_state(store: ConversationStore, env_settings: dict[str, str]) -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = SherraPipeline()
    if "api_key_override" not in st.session_state:
        st.session_state.api_key_override = ""
    if "current_session_id" not in st.session_state:
        existing = store.list_sessions()
        if existing:
            st.session_state.current_session_id = existing[0].id
        else:
            fresh = ChatSession()
            store.save(fresh)
            st.session_state.current_session_id = fresh.id
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "env_settings" not in st.session_state:
        st.session_state.env_settings = env_settings


def _get_active_session(store: ConversationStore) -> ChatSession:
    session = store.load(st.session_state.current_session_id)
    if session is None:
        session = ChatSession()
        store.save(session)
        st.session_state.current_session_id = session.id
    return session


def _trace_from_context(ctx: PipelineContext) -> dict[str, Any]:
    return {
        "model": ctx.config.model,
        "usage": asdict(ctx.usage),
        "stages": [asdict(log) for log in ctx.stage_logs],
        "payload_messages": ctx.messages,
    }


def _render_trace(trace: dict[str, Any]) -> None:
    usage = trace.get("usage", {})
    stages = trace.get("stages", [])
    model = trace.get("model", "")
    summary = (
        f"Pipeline Trace · `{model}` · "
        f"{usage.get('total_tokens', 0)} tokens · "
        f"{usage.get('latency_ms', 0):.0f} ms · "
        f"${usage.get('estimated_cost_usd', 0.0):.6f}"
    )
    with st.expander(summary, expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prompt Tokens", usage.get("prompt_tokens", 0))
        c2.metric("Completion Tokens", usage.get("completion_tokens", 0))
        c3.metric("Latency (ms)", f"{usage.get('latency_ms', 0):.1f}")
        c4.metric("Est. Cost", f"${usage.get('estimated_cost_usd', 0.0):.6f}")

        if stages:
            st.markdown("**Stage Execution**")
            for stage in stages:
                st.markdown(
                    f"- **{stage['stage_name']}** — `{stage['duration_ms']} ms` "
                    f"({stage['status']}) — {stage['detail']}"
                )

        payload = trace.get("payload_messages")
        if payload:
            st.markdown("**Assembled Prompt Payload**")
            st.json(payload)


def _render_sidebar(
    store: ConversationStore,
    session: ChatSession,
    env_settings: dict[str, str],
) -> PipelineConfig:
    with st.sidebar:
        st.markdown(f"## ✨ {APP_NAME}")
        st.caption("Pure Python LLM Playground & Pipeline")

        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            new_chat = ChatSession(model=session.model, system_prompt=session.system_prompt)
            store.save(new_chat)
            st.session_state.current_session_id = new_chat.id
            st.rerun()

        st.divider()
        st.markdown("### 💬 Conversations")
        sessions = store.list_sessions()
        for item in sessions[:20]:
            is_active = item.id == session.id
            label = f"▸ {item.title}" if is_active else item.title
            col_open, col_del = st.columns([5, 1])
            with col_open:
                if st.button(
                    label,
                    key=f"open_{item.id}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    st.session_state.current_session_id = item.id
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{item.id}", help="Delete conversation"):
                    store.delete(item.id)
                    remaining = store.list_sessions()
                    if remaining:
                        st.session_state.current_session_id = remaining[0].id
                    else:
                        replacement = ChatSession()
                        store.save(replacement)
                        st.session_state.current_session_id = replacement.id
                    st.rerun()

        st.divider()
        st.markdown("### ⚙️ Playground & Pipeline")

        model_options = list(MODEL_CATALOG.keys()) + ["Custom..."]
        default_index = (
            model_options.index(session.model)
            if session.model in MODEL_CATALOG
            else len(model_options) - 1
        )
        selected_model_choice = st.selectbox(
            "Model",
            options=model_options,
            index=default_index,
            format_func=lambda m: MODEL_CATALOG[m]["label"] if m in MODEL_CATALOG else m,
        )
        if selected_model_choice == "Custom...":
            model_name = st.text_input(
                "Custom Model ID",
                value=session.model if session.model not in MODEL_CATALOG else DEFAULT_MODEL,
            ).strip() or DEFAULT_MODEL
        else:
            model_name = selected_model_choice

        preset_options = list(PERSONA_PRESETS.keys()) + ["Custom"]
        matching_preset = next(
            (name for name, text in PERSONA_PRESETS.items() if text == session.system_prompt),
            "Custom",
        )
        selected_preset = st.selectbox(
            "Persona Preset",
            options=preset_options,
            index=preset_options.index(matching_preset),
        )
        default_sys_prompt = (
            PERSONA_PRESETS[selected_preset]
            if selected_preset in PERSONA_PRESETS
            else session.system_prompt
        )
        system_prompt = st.text_area(
            "System Prompt",
            value=default_sys_prompt,
            height=110,
        )

        model_meta = MODEL_CATALOG.get(model_name, {})
        supports_temp = model_meta.get("supports_temperature", True)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
            disabled=not supports_temp,
            help="Disabled for reasoning models (o3/o4 series).",
        )
        top_p = st.slider(
            "Top-p",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.05,
            disabled=not supports_temp,
        )
        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=128,
            max_value=4096,
            value=1024,
            step=128,
        )
        max_history = st.slider(
            "Context Window (messages)",
            min_value=0,
            max_value=50,
            value=20,
            step=2,
            help="How many recent messages the Prompt Builder stage keeps.",
        )
        stream_enabled = st.toggle("Stream responses", value=True)

        with st.expander("🔑 API Key & Usage", expanded=False):
            has_env_key = bool(env_settings.get("api_key"))
            if has_env_key:
                st.success("Loaded OPENAI_API_KEY from `.env`")
            else:
                st.warning("No OPENAI_API_KEY found in `.env`")

            override = st.text_input(
                "Override API Key (optional)",
                value=st.session_state.api_key_override,
                type="password",
                placeholder="sk-...",
            )
            st.session_state.api_key_override = override
            usage_url = env_settings.get("usage_url", "")
            if usage_url:
                st.link_button("Open OpenAI Usage Dashboard", usage_url, use_container_width=True)

        if session.messages:
            st.download_button(
                "⬇ Export Chat (.md)",
                data=ConversationStore.export_markdown(session),
                file_name=f"sherra_ai_{session.id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    effective_key = st.session_state.api_key_override.strip() or env_settings.get("api_key", "")
    return PipelineConfig(
        model=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_history_messages=max_history,
        stream=stream_enabled,
        api_key=effective_key,
        project_id=env_settings.get("project_id", ""),
    )


def _execute_turn(
    prompt: str,
    store: ConversationStore,
    session: ChatSession,
    config: PipelineConfig,
) -> None:
    pipeline: SherraPipeline = st.session_state.pipeline
    history_before = [
        {"role": m["role"], "content": m["content"]} for m in session.messages
    ]

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    session.model = config.model
    session.system_prompt = config.system_prompt
    session.add_message("user", prompt)
    store.save(session)

    with st.chat_message("assistant", avatar="✨"):
        try:
            if config.stream:
                ctx, token_iter = pipeline.run_stream(
                    user_input=prompt,
                    history=history_before,
                    config=config,
                )
                reply_text = st.write_stream(token_iter)
                if not isinstance(reply_text, str):
                    reply_text = ctx.response_text
            else:
                with st.spinner("Sherra AI pipeline running..."):
                    ctx = pipeline.run(
                        user_input=prompt,
                        history=history_before,
                        config=config,
                    )
                    reply_text = ctx.response_text
                    st.markdown(reply_text)

            trace = _trace_from_context(ctx)
            _render_trace(trace)

            session.add_message("assistant", reply_text, trace=trace)
            session.total_tokens += ctx.usage.total_tokens
            session.total_cost_usd = round(
                session.total_cost_usd + ctx.usage.estimated_cost_usd, 6
            )
            store.save(session)

        except AuthenticationError:
            st.error(
                "Authentication failed: OpenAI rejected the API key. "
                "Verify `OPENAI_API_KEY` in `.env` or enter a valid key in the sidebar."
            )
        except RateLimitError as exc:
            st.error(f"OpenAI rate limit or quota exceeded: {exc}")
        except APIError as exc:
            st.error(f"OpenAI API error: {exc}")
        except ValueError as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} — LLM Playground",
        page_icon="✨",
        layout="wide",
    )

    store = ConversationStore()
    env_settings = load_env_settings()
    _init_state(store, env_settings)
    session = _get_active_session(store)
    config = _render_sidebar(store, session, env_settings)
    pipeline: SherraPipeline = st.session_state.pipeline

    # Top header bar
    h_left, h_right = st.columns([3, 2])
    with h_left:
        st.markdown(f"## ✨ {APP_NAME}")
        st.caption(
            "Pipeline: " + "  →  ".join(f"`{name}`" for name in pipeline.stage_names())
        )
    with h_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", config.model)
        m2.metric("Session Tokens", f"{session.total_tokens:,}")
        m3.metric("Est. Cost", f"${session.total_cost_usd:.5f}")

    st.divider()

    # Empty state with starter prompts
    if not session.messages:
        st.markdown(
            f"### Welcome to {APP_NAME} 👋\n"
            "Ask anything below, or pick a starter prompt to test the Python LLM pipeline:"
        )
        cols = st.columns(2)
        for idx, starter in enumerate(STARTER_PROMPTS):
            with cols[idx % 2]:
                if st.button(starter, key=f"starter_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = starter
                    st.rerun()

    # Render conversation history
    for msg in session.messages:
        avatar = "🧑" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                _render_trace(msg["trace"])

    user_prompt = st.chat_input(f"Message {APP_NAME}...")
    if not user_prompt and st.session_state.pending_prompt:
        user_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_prompt:
        _execute_turn(user_prompt, store, session, config)


if __name__ == "__main__":
    main()
