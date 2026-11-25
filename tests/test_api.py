"""Tests for the Vercel Starlette ASGI entrypoint in api/index.py."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from starlette.testclient import TestClient

import api.index as vercel_api
from sherra_ai import OpenAIInferenceStage, SherraPipeline


class _FakeCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi "))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Vercel!"))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(
                            prompt_tokens=10, completion_tokens=4, total_tokens=14
                        ),
                    ),
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Direct Vercel reply"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class TestVercelApi(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_pipeline = vercel_api.pipeline
        vercel_api.pipeline = SherraPipeline(
            inference_stage=OpenAIInferenceStage(client_factory=lambda _cfg: _FakeClient())
        )
        self.client = TestClient(vercel_api.app)

    def tearDown(self) -> None:
        vercel_api.pipeline = self._orig_pipeline

    def test_homepage_and_health(self) -> None:
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("Sherra AI", home.text)

        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        cfg = self.client.get("/api/config")
        self.assertEqual(cfg.status_code, 200)
        self.assertEqual(len(cfg.json()["pipeline_stages"]), 4)

    def test_chat_non_streaming(self) -> None:
        resp = self.client.post(
            "/api/chat",
            json={"prompt": "Hello", "stream": False, "api_key": "sk-test"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["response_text"], "Direct Vercel reply")
        self.assertEqual(data["trace"]["usage"]["total_tokens"], 15)

    def test_chat_streaming_sse(self) -> None:
        resp = self.client.post(
            "/api/chat",
            json={"prompt": "Hello", "stream": True, "api_key": "sk-test"},
        )
        self.assertEqual(resp.status_code, 200)
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        deltas = [ev["content"] for ev in events if ev["type"] == "delta"]
        done = [ev for ev in events if ev["type"] == "done"]
        self.assertEqual("".join(deltas), "Hi Vercel!")
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["trace"]["usage"]["total_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
