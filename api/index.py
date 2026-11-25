"""Vercel Serverless ASGI entrypoint for Sherra AI (pure Python + Starlette)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure repository root is importable when Vercel loads `api/index.py`.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai import APIError, AuthenticationError, RateLimitError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from sherra_ai import (
    APP_NAME,
    DEFAULT_MODEL,
    MODEL_CATALOG,
    PERSONA_PRESETS,
    PipelineConfig,
    PipelineContext,
    SherraPipeline,
    load_env_settings,
)

pipeline = SherraPipeline()


def _trace_from_context(ctx: PipelineContext) -> dict[str, Any]:
    return {
        "model": ctx.config.model,
        "usage": asdict(ctx.usage),
        "stages": [asdict(log) for log in ctx.stage_logs],
        "payload_messages": ctx.messages,
    }


def _build_html() -> str:
    models_json = json.dumps(MODEL_CATALOG)
    personas_json = json.dumps(PERSONA_PRESETS)
    stages_json = json.dumps(pipeline.stage_names())
    env_settings = load_env_settings()
    usage_url = json.dumps(env_settings.get("usage_url", ""))
    has_server_key = json.dumps(bool(env_settings.get("api_key")))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{APP_NAME} — LLM Playground</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --sidebar: #111827;
      --panel: #161f30;
      --border: #263248;
      --text: #f3f4f6;
      --muted: #9ca3af;
      --accent: #10a37f;
      --accent-hover: #0e8e6f;
      --user-bubble: #1f2937;
      --error-bg: rgba(239, 68, 68, 0.14);
      --error-border: rgba(239, 68, 68, 0.45);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    .sidebar {{
      width: 310px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 16px;
      gap: 12px;
      overflow-y: auto;
    }}
    .brand {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 1.2rem;
      font-weight: 700;
    }}
    .badge {{
      font-size: 0.72rem;
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 2px 7px;
      border-radius: 999px;
      font-weight: 600;
    }}
    button {{
      cursor: pointer;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 8px;
      padding: 9px 12px;
      font-size: 0.88rem;
      transition: background 0.15s ease;
    }}
    button:hover {{ background: #1e293b; }}
    .btn-primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      font-weight: 600;
    }}
    .btn-primary:hover {{ background: var(--accent-hover); }}
    .section-title {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-top: 6px;
    }}
    .chat-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 170px;
      overflow-y: auto;
    }}
    .chat-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .chat-item button.title {{
      flex: 1;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 7px 10px;
    }}
    .chat-item button.active {{
      border-color: var(--accent);
      background: rgba(16, 163, 127, 0.14);
    }}
    .chat-item button.del {{
      padding: 7px 9px;
      color: var(--muted);
    }}
    label {{
      display: flex;
      flex-direction: column;
      gap: 5px;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    select, textarea, input[type="text"], input[type="password"] {{
      width: 100%;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 0.86rem;
      font-family: inherit;
    }}
    textarea {{ resize: vertical; min-height: 76px; }}
    .row-inline {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .main {{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .topbar {{
      padding: 14px 22px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      background: rgba(17, 24, 39, 0.75);
    }}
    .pipeline-bar {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 4px;
    }}
    .pipeline-bar code {{
      background: var(--panel);
      padding: 2px 6px;
      border-radius: 5px;
      color: #d1d5db;
    }}
    .metrics {{
      display: flex;
      gap: 16px;
      font-size: 0.84rem;
    }}
    .metric span {{ color: var(--muted); display: block; font-size: 0.72rem; }}
    .messages {{
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .welcome {{
      max-width: 720px;
      margin: 40px auto;
      text-align: center;
    }}
    .starters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .msg {{
      max-width: 820px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }}
    .avatar {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: var(--panel);
      border: 1px solid var(--border);
      flex-shrink: 0;
      font-size: 0.95rem;
    }}
    .msg.assistant .avatar {{
      background: rgba(16, 163, 127, 0.16);
      border-color: var(--accent);
    }}
    .bubble {{
      flex: 1;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .msg.user .bubble {{ background: var(--user-bubble); }}
    .msg.error .bubble {{
      background: var(--error-bg);
      border-color: var(--error-border);
      color: #fecaca;
    }}
    details.trace {{
      margin-top: 10px;
      background: rgba(11, 15, 23, 0.7);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 0.8rem;
      color: var(--muted);
    }}
    details.trace summary {{ cursor: pointer; color: #d1d5db; }}
    details.trace pre {{
      overflow-x: auto;
      white-space: pre-wrap;
      margin: 8px 0 0;
      color: #cbd5e1;
      font-size: 0.76rem;
    }}
    .composer {{
      padding: 16px 24px 20px;
      border-top: 1px solid var(--border);
      background: var(--sidebar);
    }}
    .composer-inner {{
      max-width: 820px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
    }}
    .composer-inner input {{
      flex: 1;
      padding: 12px 14px;
      font-size: 0.95rem;
    }}
    @media (max-width: 820px) {{
      body {{ flex-direction: column; height: auto; min-height: 100vh; overflow: auto; }}
      .sidebar {{ width: 100%; border-right: none; border-bottom: 1px solid var(--border); }}
      .messages {{ min-height: 55vh; }}
    }}
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand">
      <span>✨ {APP_NAME}</span>
      <span class="badge">Python</span>
    </div>
    <button class="btn-primary" id="newChatBtn">＋ New Chat</button>

    <div class="section-title">Conversations</div>
    <div class="chat-list" id="chatList"></div>

    <div class="section-title">Playground & Pipeline</div>
    <label>Model
      <select id="modelSelect"></select>
    </label>
    <label id="customModelWrap" style="display:none;">Custom Model ID
      <input type="text" id="customModelInput" value="{DEFAULT_MODEL}" />
    </label>
    <label>Persona Preset
      <select id="personaSelect"></select>
    </label>
    <label>System Prompt
      <textarea id="systemPrompt"></textarea>
    </label>
    <label>
      <div class="row-inline"><span>Temperature</span><strong id="tempVal">0.70</strong></div>
      <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7" />
    </label>
    <label>
      <div class="row-inline"><span>Top-p</span><strong id="topPVal">1.00</strong></div>
      <input type="range" id="topP" min="0.1" max="1" step="0.05" value="1.0" />
    </label>
    <label>
      <div class="row-inline"><span>Max Output Tokens</span><strong id="maxTokensVal">1024</strong></div>
      <input type="range" id="maxTokens" min="128" max="4096" step="128" value="1024" />
    </label>
    <label>
      <div class="row-inline"><span>Context Window (msgs)</span><strong id="historyVal">20</strong></div>
      <input type="range" id="maxHistory" min="0" max="50" step="2" value="20" />
    </label>
    <label class="row-inline" style="flex-direction:row;">
      <span>Stream responses</span>
      <input type="checkbox" id="streamToggle" checked />
    </label>

    <div class="section-title">API Key & Usage</div>
    <div id="keyStatus" style="font-size:0.78rem;color:var(--muted);"></div>
    <label>Override OpenAI API Key (optional)
      <input type="password" id="apiKeyOverride" placeholder="sk-..." />
    </label>
    <a id="usageLink" target="_blank" rel="noopener noreferrer" style="color:var(--accent);font-size:0.82rem;text-decoration:none;">
      ↗ Open OpenAI Usage Dashboard
    </a>
    <button id="exportBtn">⬇ Export Chat (.md)</button>
  </aside>

  <main class="main">
    <header class="topbar">
      <div>
        <strong style="font-size:1.1rem;">✨ {APP_NAME} — Python LLM Playground</strong>
        <div class="pipeline-bar" id="pipelineBar"></div>
      </div>
      <div class="metrics">
        <div class="metric"><span>Model</span><strong id="metricModel">{DEFAULT_MODEL}</strong></div>
        <div class="metric"><span>Session Tokens</span><strong id="metricTokens">0</strong></div>
        <div class="metric"><span>Est. Cost</span><strong id="metricCost">$0.00000</strong></div>
      </div>
    </header>

    <section class="messages" id="messages"></section>

    <form class="composer" id="chatForm">
      <div class="composer-inner">
        <input type="text" id="promptInput" placeholder="Message {APP_NAME}..." autocomplete="off" />
        <button type="submit" class="btn-primary" id="sendBtn">Send</button>
      </div>
    </form>
  </main>

  <script>
    const MODEL_CATALOG = {models_json};
    const PERSONA_PRESETS = {personas_json};
    const PIPELINE_STAGES = {stages_json};
    const USAGE_URL = {usage_url};
    const HAS_SERVER_KEY = {has_server_key};
    const STORAGE_KEY = "sherra_ai_sessions_v1";
    const STARTERS = [
      "Explain how transformer attention works in simple terms.",
      "Write a clean Python function to parse and validate JSON logs.",
      "Help me plan a 7-day study roadmap for Linear Algebra.",
      "Compare REST vs GraphQL with concrete trade-offs."
    ];

    let sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!sessions.length) {{
      sessions = [createSession()];
    }}
    let activeId = sessions[0].id;
    let isBusy = false;

    function createSession() {{
      return {{
        id: Math.random().toString(36).slice(2, 12),
        title: "New Chat",
        messages: [],
        totalTokens: 0,
        totalCost: 0
      }};
    }}

    function saveSessions() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }}

    function activeSession() {{
      return sessions.find(s => s.id === activeId) || sessions[0];
    }}

    const modelSelect = document.getElementById("modelSelect");
    const customModelWrap = document.getElementById("customModelWrap");
    const customModelInput = document.getElementById("customModelInput");
    const personaSelect = document.getElementById("personaSelect");
    const systemPromptEl = document.getElementById("systemPrompt");
    const tempEl = document.getElementById("temperature");
    const topPEl = document.getElementById("topP");
    const maxTokensEl = document.getElementById("maxTokens");
    const maxHistoryEl = document.getElementById("maxHistory");
    const streamToggle = document.getElementById("streamToggle");
    const apiKeyOverride = document.getElementById("apiKeyOverride");

    Object.entries(MODEL_CATALOG).forEach(([id, meta]) => {{
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = meta.label;
      modelSelect.appendChild(opt);
    }});
    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "Custom...";
    modelSelect.appendChild(customOpt);

    Object.keys(PERSONA_PRESETS).forEach(name => {{
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      personaSelect.appendChild(opt);
    }});
    systemPromptEl.value = PERSONA_PRESETS["Sherra Default"];

    document.getElementById("pipelineBar").innerHTML =
      "Pipeline: " + PIPELINE_STAGES.map(s => `<code>${{s}}</code>`).join(" → ");
    document.getElementById("keyStatus").textContent = HAS_SERVER_KEY
      ? "✓ Server OPENAI_API_KEY configured"
      : "⚠ Set OPENAI_API_KEY in Vercel env or paste below";
    const usageLink = document.getElementById("usageLink");
    if (USAGE_URL) usageLink.href = USAGE_URL;
    else usageLink.style.display = "none";

    function currentModel() {{
      return modelSelect.value === "__custom__"
        ? (customModelInput.value.trim() || "{DEFAULT_MODEL}")
        : modelSelect.value;
    }}

    function syncControls() {{
      const m = currentModel();
      const meta = MODEL_CATALOG[m];
      const supportsTemp = meta ? meta.supports_temperature : !/^(o1|o3|o4)/.test(m);
      tempEl.disabled = !supportsTemp;
      topPEl.disabled = !supportsTemp;
      customModelWrap.style.display = modelSelect.value === "__custom__" ? "flex" : "none";
      document.getElementById("metricModel").textContent = m;
      document.getElementById("tempVal").textContent = Number(tempEl.value).toFixed(2);
      document.getElementById("topPVal").textContent = Number(topPEl.value).toFixed(2);
      document.getElementById("maxTokensVal").textContent = maxTokensEl.value;
      document.getElementById("historyVal").textContent = maxHistoryEl.value;
    }}

    modelSelect.addEventListener("change", syncControls);
    customModelInput.addEventListener("input", syncControls);
    tempEl.addEventListener("input", syncControls);
    topPEl.addEventListener("input", syncControls);
    maxTokensEl.addEventListener("input", syncControls);
    maxHistoryEl.addEventListener("input", syncControls);
    personaSelect.addEventListener("change", () => {{
      systemPromptEl.value = PERSONA_PRESETS[personaSelect.value] || "";
    }});

    function renderChatList() {{
      const list = document.getElementById("chatList");
      list.innerHTML = "";
      sessions.forEach(s => {{
        const row = document.createElement("div");
        row.className = "chat-item";
        const btn = document.createElement("button");
        btn.className = "title" + (s.id === activeId ? " active" : "");
        btn.textContent = s.title;
        btn.onclick = () => {{ activeId = s.id; render(); }};
        const del = document.createElement("button");
        del.className = "del";
        del.textContent = "🗑";
        del.onclick = () => {{
          sessions = sessions.filter(x => x.id !== s.id);
          if (!sessions.length) sessions = [createSession()];
          if (!sessions.some(x => x.id === activeId)) activeId = sessions[0].id;
          saveSessions();
          render();
        }};
        row.append(btn, del);
        list.appendChild(row);
      }});
    }}

    function createTraceDetails(trace) {{
      if (!trace) return null;
      const details = document.createElement("details");
      details.className = "trace";
      const u = trace.usage || {{}};
      const summary = document.createElement("summary");
      summary.textContent =
        `Pipeline Trace · ${{trace.model}} · ${{u.total_tokens || 0}} tokens · ` +
        `${{Math.round(u.latency_ms || 0)}} ms · $${{Number(u.estimated_cost_usd || 0).toFixed(6)}}`;
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(trace, null, 2);
      details.append(summary, pre);
      return details;
    }}

    function appendMessageEl(msg) {{
      const container = document.getElementById("messages");
      const row = document.createElement("div");
      row.className = `msg ${{msg.role}}${{msg.error ? " error" : ""}}`;
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = msg.role === "user" ? "🧑" : "✨";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = msg.content;
      if (msg.trace) {{
        const t = createTraceDetails(msg.trace);
        if (t) bubble.appendChild(t);
      }}
      row.append(avatar, bubble);
      container.appendChild(row);
      container.scrollTop = container.scrollHeight;
      return bubble;
    }}

    function renderMessages() {{
      const s = activeSession();
      const container = document.getElementById("messages");
      container.innerHTML = "";
      document.getElementById("metricTokens").textContent = (s.totalTokens || 0).toLocaleString();
      document.getElementById("metricCost").textContent = `$${{Number(s.totalCost || 0).toFixed(5)}}`;

      if (!s.messages.length) {{
        const wrap = document.createElement("div");
        wrap.className = "welcome";
        wrap.innerHTML = `<h2>Welcome to {APP_NAME} 👋</h2><p style="color:var(--muted);">Ask anything below, or pick a starter prompt:</p>`;
        const grid = document.createElement("div");
        grid.className = "starters";
        STARTERS.forEach(text => {{
          const btn = document.createElement("button");
          btn.textContent = text;
          btn.onclick = () => sendPrompt(text);
          grid.appendChild(btn);
        }});
        wrap.appendChild(grid);
        container.appendChild(wrap);
        return;
      }}
      s.messages.forEach(appendMessageEl);
    }}

    function render() {{
      syncControls();
      renderChatList();
      renderMessages();
    }}

    document.getElementById("newChatBtn").onclick = () => {{
      const s = createSession();
      sessions.unshift(s);
      activeId = s.id;
      saveSessions();
      render();
    }};

    document.getElementById("exportBtn").onclick = () => {{
      const s = activeSession();
      const lines = [`# ${{s.title}}`, ""];
      s.messages.forEach(m => {{
        lines.push(`### ${{m.role === "user" ? "User" : "{APP_NAME}"}}`, "", m.content, "");
      }});
      const blob = new Blob([lines.join("\\n")], {{ type: "text/markdown" }});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `sherra_ai_${{s.id}}.md`;
      a.click();
    }};

    async function sendPrompt(text) {{
      if (isBusy || !text.trim()) return;
      isBusy = true;
      document.getElementById("sendBtn").disabled = true;

      const s = activeSession();
      const historyBefore = s.messages
        .filter(m => !m.error)
        .map(m => ({{ role: m.role, content: m.content }}));

      if (s.title === "New Chat") {{
        s.title = text.length > 42 ? text.slice(0, 42) + "..." : text;
      }}
      s.messages.push({{ role: "user", content: text }});
      saveSessions();
      renderChatList();
      renderMessages();

      const assistantBubble = appendMessageEl({{ role: "assistant", content: "Thinking..." }});
      const payload = {{
        prompt: text,
        history: historyBefore,
        model: currentModel(),
        system_prompt: systemPromptEl.value,
        temperature: Number(tempEl.value),
        top_p: Number(topPEl.value),
        max_tokens: Number(maxTokensEl.value),
        max_history_messages: Number(maxHistoryEl.value),
        stream: streamToggle.checked,
        api_key: apiKeyOverride.value.trim()
      }};

      try {{
        const resp = await fetch("/api/chat", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});

        if (!payload.stream) {{
          const data = await resp.json();
          if (!resp.ok) throw new Error(data.error || "Request failed");
          assistantBubble.textContent = data.response_text;
          const t = createTraceDetails(data.trace);
          if (t) assistantBubble.appendChild(t);
          s.messages.push({{ role: "assistant", content: data.response_text, trace: data.trace }});
          s.totalTokens += data.trace?.usage?.total_tokens || 0;
          s.totalCost += data.trace?.usage?.estimated_cost_usd || 0;
        }} else {{
          if (!resp.ok) {{
            const errData = await resp.json().catch(() => ({{}}));
            throw new Error(errData.error || `HTTP ${{resp.status}}`);
          }}
          const reader = resp.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          let fullText = "";
          let finalTrace = null;
          assistantBubble.textContent = "";

          while (true) {{
            const {{ value, done }} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {{ stream: true }});
            const events = buffer.split("\\n\\n");
            buffer = events.pop() || "";
            for (const ev of events) {{
              if (!ev.startsWith("data: ")) continue;
              const parsed = JSON.parse(ev.slice(6));
              if (parsed.type === "delta") {{
                fullText += parsed.content;
                assistantBubble.textContent = fullText;
              }} else if (parsed.type === "done") {{
                finalTrace = parsed.trace;
                fullText = parsed.response_text || fullText;
                assistantBubble.textContent = fullText;
                const t = createTraceDetails(finalTrace);
                if (t) assistantBubble.appendChild(t);
              }} else if (parsed.type === "error") {{
                throw new Error(parsed.message);
              }}
            }}
          }}
          s.messages.push({{ role: "assistant", content: fullText, trace: finalTrace }});
          s.totalTokens += finalTrace?.usage?.total_tokens || 0;
          s.totalCost += finalTrace?.usage?.estimated_cost_usd || 0;
        }}
        saveSessions();
        document.getElementById("metricTokens").textContent = (s.totalTokens || 0).toLocaleString();
        document.getElementById("metricCost").textContent = `$${{Number(s.totalCost || 0).toFixed(5)}}`;
      }} catch (err) {{
        assistantBubble.parentElement.classList.add("error");
        assistantBubble.textContent = err.message || String(err);
      }} finally {{
        isBusy = false;
        document.getElementById("sendBtn").disabled = false;
      }}
    }}

    document.getElementById("chatForm").addEventListener("submit", (e) => {{
      e.preventDefault();
      const input = document.getElementById("promptInput");
      const val = input.value.trim();
      if (!val) return;
      input.value = "";
      sendPrompt(val);
    }});

    render();
  </script>
</body>
</html>"""


async def homepage(_request: Request) -> HTMLResponse:
    return HTMLResponse(_build_html())


async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "app": APP_NAME})


async def get_config(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "app": APP_NAME,
            "default_model": DEFAULT_MODEL,
            "models": MODEL_CATALOG,
            "personas": PERSONA_PRESETS,
            "pipeline_stages": pipeline.stage_names(),
        }
    )


def _format_openai_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, AuthenticationError):
        return 401, "Authentication failed: OpenAI rejected the API key."
    if isinstance(exc, RateLimitError):
        return 429, f"OpenAI rate limit or quota exceeded: {exc}"
    if isinstance(exc, APIError):
        return 502, f"OpenAI API error: {exc}"
    return 400, str(exc)


async def chat(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    prompt = str(body.get("prompt", ""))
    history = body.get("history") or []
    env_settings = load_env_settings()
    api_key = str(body.get("api_key") or "").strip() or env_settings.get("api_key", "")

    config = PipelineConfig(
        model=str(body.get("model") or DEFAULT_MODEL),
        system_prompt=str(body.get("system_prompt") or PERSONA_PRESETS["Sherra Default"]),
        temperature=float(body.get("temperature", 0.7)),
        top_p=float(body.get("top_p", 1.0)),
        max_tokens=int(body.get("max_tokens", 1024)),
        max_history_messages=int(body.get("max_history_messages", 20)),
        stream=bool(body.get("stream", True)),
        api_key=api_key,
        project_id=env_settings.get("project_id", ""),
    )

    if not config.stream:
        try:
            ctx = pipeline.run(user_input=prompt, history=history, config=config)
            return JSONResponse(
                {"response_text": ctx.response_text, "trace": _trace_from_context(ctx)}
            )
        except (AuthenticationError, RateLimitError, APIError, ValueError) as exc:
            status_code, msg = _format_openai_error(exc)
            return JSONResponse({"error": msg}, status_code=status_code)

    try:
        ctx, token_iter = pipeline.run_stream(
            user_input=prompt, history=history, config=config
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    def _sse_stream():
        try:
            for chunk in token_iter:
                yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "response_text": ctx.response_text,
                        "trace": _trace_from_context(ctx),
                    }
                )
                + "\n\n"
            )
        except (AuthenticationError, RateLimitError, APIError, ValueError) as exc:
            _, msg = _format_openai_error(exc)
            yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app = Starlette(
    routes=[
        Route("/", homepage, methods=["GET"]),
        Route("/api/health", health, methods=["GET"]),
        Route("/api/config", get_config, methods=["GET"]),
        Route("/api/chat", chat, methods=["POST"]),
    ]
)
