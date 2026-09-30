"""Regenerate the README screenshots in docs/images/ (offline, reproducible).

    pip install -e ".[dev]" ansi2html nbconvert playwright
    python scripts/make_screenshots.py

Every terminal screenshot is the *real* output of the command shown in its title bar,
run with the offline provider so no API key is needed.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "FORCE_COLOR": "1", "PY_COLORS": "1", "LLM_PROVIDER": "offline", "COLUMNS": "110"}

TERMINAL_CSS = """
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.win{display:inline-block;min-width:860px;max-width:1180px;border-radius:10px;overflow:hidden;
     box-shadow:0 12px 32px rgba(15,23,42,.28);background:#0f172a}
.bar{background:#1e293b;padding:10px 14px;display:flex;align-items:center;gap:8px}
.dot{width:12px;height:12px;border-radius:50%}
.title{color:#94a3b8;font-size:13px;margin-left:10px;font-family:ui-monospace,Consolas,monospace}
pre{margin:0;padding:18px 22px;color:#e2e8f0;font:14px/1.55 ui-monospace,'Cascadia Code',Consolas,monospace;
    white-space:pre-wrap;word-break:break-word}
.prompt{color:#22c55e}
"""


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def terminal_html(command: str, output: str, title: str) -> str:
    conv = Ansi2HTMLConverter(inline=True, dark_bg=True)
    body = conv.convert(output, full=False)
    return f"""<html><head><meta charset="utf-8"><style>{TERMINAL_CSS}</style></head><body>
<div class="win"><div class="bar"><span class="dot" style="background:#ef4444"></span>
<span class="dot" style="background:#f59e0b"></span><span class="dot" style="background:#22c55e"></span>
<span class="title">{html.escape(title)}</span></div>
<pre><span class="prompt">$</span> {html.escape(command)}\n{body}</pre></div></body></html>"""


def shoot(page, html_doc: str, name: str, selector: str = ".win") -> None:
    page.set_content(html_doc, wait_until="networkidle")
    page.locator(selector).first.screenshot(path=str(OUT / name))
    print("  ✓", name)


ARCHITECTURE = """<html><head><meta charset="utf-8"><style>
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.card{display:inline-block;background:#fff;border-radius:12px;padding:22px 26px;box-shadow:0 12px 32px rgba(15,23,42,.18)}
h2{margin:0 0 4px;font-size:20px;color:#0f172a}p{margin:0 0 14px;color:#475569;font-size:14px}
text{font-family:-apple-system,Segoe UI,sans-serif} .mono text{font-family:ui-monospace,Consolas,monospace}
</style></head><body><div class="card">
<h2>How the agent works: the tool-calling loop</h2>
<p><code>create_agent(model, tools, system_prompt)</code> builds a LangGraph graph that loops until the model stops asking for tools.</p>
<svg width="1060" height="380" viewBox="0 0 1060 380" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<rect x="10" y="150" width="170" height="80" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
<text x="95" y="183" text-anchor="middle" font-size="16" font-weight="700" fill="#075985">User question</text>
<text x="95" y="206" text-anchor="middle" font-size="13" fill="#0c4a6e">"Add 25 and 15,</text>
<text x="95" y="222" text-anchor="middle" font-size="13" fill="#0c4a6e">then multiply by 2"</text>
<rect x="260" y="130" width="220" height="120" rx="12" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<text x="370" y="165" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">LLM (chat model)</text>
<text x="370" y="190" text-anchor="middle" font-size="13" fill="#78350f">OpenAI · watsonx Granite</text>
<text x="370" y="210" text-anchor="middle" font-size="13" fill="#78350f">· offline rule-based demo</text>
<text x="370" y="234" text-anchor="middle" font-size="12" fill="#92400e">reads tool names + docstrings</text>
<rect x="640" y="20" width="410" height="200" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="845" y="50" text-anchor="middle" font-size="16" font-weight="700" fill="#6b21a8">Tools (@tool)</text>
<g class="mono" font-size="12.5" fill="#581c87" font-family="ui-monospace,Consolas,monospace" style="white-space:pre" xml:space="preserve">
<text x="660" y="82">add_numbers        "25, 15" → {"result": 40}</text>
<text x="660" y="106">subtract_numbers   "100, 20, 10" → 70</text>
<text x="660" y="130">multiply_numbers   "40, 2" → {"result": 80}</text>
<text x="660" y="154">divide_numbers     "10, 0" → "Cannot divide by zero."</text>
<text x="660" y="178">power_tool         "2^8" → 256</text>
<text x="660" y="202">search_wikipedia   optional, needs network</text></g>
<rect x="640" y="270" width="410" height="90" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="845" y="302" text-anchor="middle" font-size="16" font-weight="700" fill="#166534">Final answer</text>
<text x="845" y="328" text-anchor="middle" font-size="13" fill="#14532d">"(25 + 15) × 2 = 80"</text>
<text x="845" y="348" text-anchor="middle" font-size="12" fill="#166534">response["messages"][-1].content</text>
<line x1="180" y1="190" x2="256" y2="190" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<path d="M480,160 C540,150 580,120 636,110" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="488" y="112" font-size="12" fill="#334155">1 · AIMessage.tool_calls</text>
<path d="M640,180 C580,200 540,215 484,215" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="500" y="244" font-size="12" fill="#334155">2 · ToolMessage (result)</text>
<text x="500" y="260" font-size="11" fill="#64748b">repeat per step</text>
<path d="M430,250 C480,320 560,320 636,318" fill="none" stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>
<text x="420" y="340" font-size="12" fill="#166534">3 · no more tool calls → answer</text>
</svg></div></body></html>"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)

        shoot(page, ARCHITECTURE, "architecture.png", ".card")

        shots = [
            ("tools-direct.png", "Tools called directly (no LLM)",
             f"{sys.executable} examples/tools_demo.py", "python examples/tools_demo.py"),
            ("agent-trace-power.png", "Agent trace: the power tool exercise",
             'math-assistant "Calculate 5 to the power of 2." --provider offline --trace', None),
            ("agent-trace-chained.png", "Agent trace: chaining two tools",
             'math-assistant "Add 25 and 15, then multiply by 2" -p offline -t', None),
            ("agent-error-handling.png", "Agent trace: tool errors are returned, not raised",
             'math-assistant "Divide 10 by 0" -p offline -t && math-assistant "Who wrote Hamlet?" -p offline -t', None),
            ("quickstart.png", "Quickstart example",
             f"{sys.executable} examples/quickstart.py offline", "python examples/quickstart.py offline"),
            ("pytest-results.png", "Test suite",
             f"{sys.executable} -m pytest -v --color=yes -p no:cacheprovider", "pytest -v"),
            ("pytest-coverage.png", "Coverage",
             f"{sys.executable} -m pytest -q --color=yes -p no:cacheprovider --cov=math_assistant --cov-report=term-missing",
             "pytest --cov=math_assistant --cov-report=term-missing"),
        ]
        for name, title, cmd, shown in shots:
            shoot(page, terminal_html(shown or cmd, run(cmd), title), name)

        # Executed walkthrough notebook rendered as HTML
        nb_html = run(f"{sys.executable} -m jupyter nbconvert --to html --stdout notebooks/walkthrough.ipynb 2>/dev/null")
        page.set_content(nb_html, wait_until="networkidle")
        page.set_viewport_size({"width": 1100, "height": 900})
        page.screenshot(path=str(OUT / "notebook-walkthrough.png"), full_page=True)
        print("  ✓ notebook-walkthrough.png")
        browser.close()


if __name__ == "__main__":
    main()
