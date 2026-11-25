"""Pure Python terminal CLI for Sherra AI using the same SherraPipeline."""

from __future__ import annotations

import argparse
import sys

from openai import APIError, AuthenticationError, RateLimitError

from sherra_ai import (
    APP_NAME,
    DEFAULT_MODEL,
    PERSONA_PRESETS,
    ChatSession,
    ConversationStore,
    PipelineConfig,
    SherraPipeline,
    load_env_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} — Pure Python Terminal Chat & LLM Pipeline"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model ID")
    parser.add_argument(
        "--persona",
        default="Sherra Default",
        choices=list(PERSONA_PRESETS.keys()),
        help="System prompt preset",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = load_env_settings()
    if not env["api_key"]:
        print("Error: OPENAI_API_KEY not found in .env or environment.", file=sys.stderr)
        return 1

    pipeline = SherraPipeline()
    store = ConversationStore()
    session = ChatSession(
        model=args.model,
        system_prompt=PERSONA_PRESETS[args.persona],
    )
    config = PipelineConfig(
        model=args.model,
        system_prompt=PERSONA_PRESETS[args.persona],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=not args.no_stream,
        api_key=env["api_key"],
        project_id=env["project_id"],
    )

    print(f"\n✨ {APP_NAME} CLI | model={config.model} | persona={args.persona}")
    print("Pipeline: " + " -> ".join(pipeline.stage_names()))
    print("Commands: /new (reset chat), /usage (session stats), /quit (exit)\n")

    while True:
        try:
            user_text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            print("Goodbye!")
            return 0
        if user_text == "/new":
            session = ChatSession(model=config.model, system_prompt=config.system_prompt)
            print("Started a new Sherra AI conversation.\n")
            continue
        if user_text == "/usage":
            print(
                f"Session tokens: {session.total_tokens} | "
                f"Est. cost: ${session.total_cost_usd:.6f}\n"
            )
            continue

        history = [{"role": m["role"], "content": m["content"]} for m in session.messages]
        session.add_message("user", user_text)

        try:
            print(f"{APP_NAME} > ", end="", flush=True)
            if config.stream:
                ctx, token_iter = pipeline.run_stream(user_text, history=history, config=config)
                for chunk in token_iter:
                    print(chunk, end="", flush=True)
                print()
            else:
                ctx = pipeline.run(user_text, history=history, config=config)
                print(ctx.response_text)

            session.add_message("assistant", ctx.response_text)
            session.total_tokens += ctx.usage.total_tokens
            session.total_cost_usd = round(
                session.total_cost_usd + ctx.usage.estimated_cost_usd, 6
            )
            store.save(session)
            print(
                f"  [tokens: {ctx.usage.total_tokens} | "
                f"latency: {ctx.usage.latency_ms:.0f} ms | "
                f"est: ${ctx.usage.estimated_cost_usd:.6f}]\n"
            )
        except AuthenticationError:
            print("\nError: OpenAI rejected your API key.\n", file=sys.stderr)
        except (RateLimitError, APIError, ValueError) as exc:
            print(f"\nError: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
