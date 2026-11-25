# Sherra AI — Python LLM Playground & ChatGPT-Style Bot

A pure-Python LLM Playground and ChatGPT-style assistant (**Sherra AI**) powered by the OpenAI API and a modular 4-stage Python pipeline:

1. **Input Preprocessor** (`InputPreprocessorStage`) — validates and normalizes user prompts.
2. **Context & Prompt Builder** (`PromptBuilderStage`) — injects the system persona and trims chat history to your configured context window.
3. **OpenAI LLM Inference** (`OpenAIInferenceStage`) — calls OpenAI Chat Completions with optional live token streaming and usage tracking.
4. **Response Postprocessor** (`ResponsePostprocessorStage`) — finalizes text, computes latency, and estimates token cost.

## Project Entrypoints

- **`api/index.py` + `vercel.json`** — Serverless Starlette ASGI web playground for **Vercel** deployment.
- **`app.py`** — Local Streamlit web playground.
- **`cli.py`** — Local terminal chat CLI.

## Deploy on Vercel

1. Push this repository to GitHub.
2. Open [Vercel New Project](https://vercel.com/new) and import `Iadityaraman/sherra-ai`.
3. Add the environment variable `OPENAI_API_KEY` (and optionally `OPENAI_USAGE_URL`) in Vercel Project Settings.
4. Click **Deploy** — Vercel installs `requirements.txt` (`openai`, `python-dotenv`, `starlette`) and serves `api/index.py`.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Your `.env` file holds `OPENAI_API_KEY` (keep `.env` private and never commit it).

```bash
# Option A: Streamlit playground
.venv/bin/streamlit run app.py

# Option B: Vercel ASGI app locally
.venv/bin/uvicorn api.index:app --reload

# Option C: Terminal CLI
.venv/bin/python cli.py
```

## Run Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
