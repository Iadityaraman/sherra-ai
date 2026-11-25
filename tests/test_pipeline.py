"""Unit tests for Sherra AI pipeline and conversation storage."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from sherra_ai import (
    ChatSession,
    ConversationStore,
    OpenAIInferenceStage,
    PipelineConfig,
    SherraPipeline,
)


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="from Sherra!"))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(
                            prompt_tokens=15,
                            completion_tokens=5,
                            total_tokens=20,
                        ),
                    ),
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="  Sherra AI response  "))
            ],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
            ),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TestSherraPipeline(unittest.TestCase):
    def test_non_streaming_pipeline_runs_all_stages(self) -> None:
        fake_client = _FakeOpenAIClient()
        pipeline = SherraPipeline(
            inference_stage=OpenAIInferenceStage(client_factory=lambda _cfg: fake_client)
        )
        history = [
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "reply 2"},
        ]
        config = PipelineConfig(
            model="gpt-4o-mini",
            system_prompt="You are Sherra AI.",
            max_history_messages=2,
            stream=False,
        )

        ctx = pipeline.run("  What is 2+2?  ", history=history, config=config)

        self.assertEqual(ctx.normalized_input, "What is 2+2?")
        self.assertEqual(ctx.response_text, "Sherra AI response")
        self.assertEqual(len(ctx.stage_logs), 4)
        self.assertEqual(ctx.usage.total_tokens, 20)
        self.assertGreater(ctx.usage.estimated_cost_usd, 0.0)
        # System + last 2 history messages + current user input = 4 messages
        self.assertEqual(len(ctx.messages), 4)
        self.assertEqual(ctx.messages[0]["role"], "system")
        self.assertEqual(ctx.messages[1]["content"], "msg 2")
        self.assertEqual(ctx.messages[-1]["content"], "What is 2+2?")

    def test_streaming_pipeline_accumulates_text_and_usage(self) -> None:
        fake_client = _FakeOpenAIClient()
        pipeline = SherraPipeline(
            inference_stage=OpenAIInferenceStage(client_factory=lambda _cfg: fake_client)
        )
        config = PipelineConfig(model="gpt-4o-mini", stream=True)

        ctx, stream_iter = pipeline.run_stream("Hi Sherra", config=config)
        chunks = list(stream_iter)

        self.assertEqual(chunks, ["Hello ", "from Sherra!"])
        self.assertEqual(ctx.response_text, "Hello from Sherra!")
        self.assertEqual(ctx.usage.prompt_tokens, 15)
        self.assertEqual(ctx.usage.completion_tokens, 5)
        self.assertEqual(ctx.usage.total_tokens, 20)
        self.assertEqual(len(ctx.stage_logs), 4)

    def test_reasoning_model_omits_temperature_and_uses_max_completion_tokens(self) -> None:
        fake_client = _FakeOpenAIClient()
        pipeline = SherraPipeline(
            inference_stage=OpenAIInferenceStage(client_factory=lambda _cfg: fake_client)
        )
        config = PipelineConfig(model="o4-mini", max_tokens=512, stream=False)

        pipeline.run("Solve x+1=3", config=config)

        sent = fake_client.completions.last_kwargs
        self.assertNotIn("temperature", sent)
        self.assertNotIn("top_p", sent)
        self.assertEqual(sent["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", sent)

    def test_empty_input_raises_value_error(self) -> None:
        pipeline = SherraPipeline(
            inference_stage=OpenAIInferenceStage(client_factory=lambda _cfg: _FakeOpenAIClient())
        )
        with self.assertRaises(ValueError):
            pipeline.run("   ")


class TestConversationStore(unittest.TestCase):
    def test_save_load_list_export_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(tmpdir)
            session = ChatSession()
            session.add_message("user", "Hello Sherra AI")
            session.add_message("assistant", "Hi there! How can I help?")
            session.total_tokens = 42
            store.save(session)

            loaded = store.load(session.id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.title, "Hello Sherra AI")
            self.assertEqual(len(loaded.messages), 2)

            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)

            md = ConversationStore.export_markdown(loaded)
            self.assertIn("Sherra AI", md)
            self.assertIn("Hello Sherra AI", md)

            self.assertTrue(store.delete(session.id))
            self.assertIsNone(store.load(session.id))


if __name__ == "__main__":
    unittest.main()
