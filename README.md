# Sherra AI — Python LLM Playground & ChatGPT-Style Bot

A pure-Python LLM Playground and ChatGPT-style assistant (**Sherra AI**) powered by the OpenAI API and a modular 4-stage Python pipeline:

1. **Input Preprocessor** (`InputPreprocessorStage`) — validates and normalizes user prompts.
2. **Context & Prompt Builder** (`PromptBuilderStage`) — injects the system persona and trims chat history to your configured context window.
3. **OpenAI LLM Inference** (`OpenAIInferenceStage`) — calls OpenAI Chat Completions with optional live token streaming and usage tracking.
4. **Response Postprocessor** (`ResponsePostprocessorStage`) — finalizes text, computes latency, and estimates token cost.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Your `.env` file holds `OPENAI_API_KEY` (keep `.env` private and never commit it).

## Run the Web Playground (Streamlit)

```bash
.venv/bin/streamlit run app.py
```

## Run the Terminal Bot (CLI)

```bash
.venv/bin/python cli.py
```

## Run Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
