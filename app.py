import asyncio
import json
import os
import time
import html as html_lib

import streamlit as st
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

from agents.summary_agent import create_agent as create_summary_agent
from agents.action_agent import create_agent as create_action_agent
from agents.risk_agent import create_agent as create_risk_agent
from pdf_parser import pdf_to_chunks

load_dotenv()

# ── Page Config ──
st.set_page_config(page_title="Document Intelligence", page_icon="🔍", layout="wide")

# ── Full Custom Theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg: #06080f;
    --surface: #0c1220;
    --surface2: #131b2e;
    --surface3: #1a2440;
    --border: #1c2744;
    --border-light: #243052;
    --text: #e8edf5;
    --text-secondary: #94a3c0;
    --text-muted: #5c6b8a;
    --accent: #635bff;
    --accent-light: #7c75ff;
    --accent-bg: rgba(99,91,255,.08);
    --accent-border: rgba(99,91,255,.2);
    --green: #00d4aa;
    --green-bg: rgba(0,212,170,.08);
    --green-border: rgba(0,212,170,.2);
    --amber: #ffb224;
    --amber-bg: rgba(255,178,36,.06);
    --amber-border: rgba(255,178,36,.15);
    --blue: #3b82f6;
    --blue-bg: rgba(59,130,246,.08);
    --blue-border: rgba(59,130,246,.2);
    --red: #ff4d6a;
    --r: 14px;
    --r-sm: 10px;
}

/* ── Global ── */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1080px !important; }
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Hero ── */
.hero { text-align: center; padding: 1.5rem 0 2.5rem; }
.hero-chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 100px; padding: 6px 16px; margin-bottom: 16px;
}
.hero-chip .pulse {
    width: 7px; height: 7px; background: var(--green);
    border-radius: 50%; animation: glow 2s ease-in-out infinite;
}
@keyframes glow {
    0%,100% { box-shadow: 0 0 4px var(--green); opacity:1; }
    50% { box-shadow: 0 0 12px var(--green); opacity:.6; }
}
.hero-chip span {
    font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: var(--accent-light);
}
.hero h1 {
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #fff 0%, var(--accent-light) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 8px;
}
.hero p { color: var(--text-secondary); font-size: 15px; font-weight: 400; margin: 0; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--r) !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-bg) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-secondary) !important; }
[data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

/* ── Primary button ── */
.stButton > button, .stDownloadButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%) !important;
    color: #fff !important; border: none !important;
    padding: 14px 24px !important; font-weight: 700 !important;
    font-size: 15px !important; border-radius: var(--r-sm) !important;
    letter-spacing: 0.3px !important; transition: all .25s !important;
    box-shadow: 0 4px 16px rgba(99,91,255,.2) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,91,255,.35) !important;
}
.stButton > button:disabled {
    opacity: .3 !important; transform: none !important;
    box-shadow: none !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 4px 16px rgba(0,212,170,.15) !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 8px 30px rgba(0,212,170,.3) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important; color: var(--text-muted) !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 8px 20px !important; letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important; border-color: var(--accent) !important;
    color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Spinner ── */
.stSpinner > div { border-color: var(--accent) transparent transparent !important; }
.stSpinner > div > span { color: var(--text-secondary) !important; }

/* ── Agent status cards ── */
.agent-row {
    display: flex; align-items: center; gap: 14px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r-sm); padding: 14px 18px; margin-bottom: 8px;
    transition: border-color .3s;
}
.agent-row.done { border-color: var(--green-border); }
.agent-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    background: var(--text-muted);
}
.agent-dot.running { background: var(--accent); animation: glow-dot 1.5s infinite; }
.agent-dot.done { background: var(--green); }
@keyframes glow-dot {
    0%,100% { box-shadow: 0 0 6px var(--accent); }
    50% { box-shadow: 0 0 14px var(--accent); opacity: .5; }
}
.agent-name { flex: 1; font-weight: 600; font-size: 14px; color: var(--text); }
.agent-tag {
    font-size: 10px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; padding: 4px 10px; border-radius: 6px;
}
.agent-tag.running { background: var(--accent-bg); color: var(--accent-light); }
.agent-tag.done { background: var(--green-bg); color: var(--green); }

/* ── Meta bar ── */
.meta-bar {
    display: flex; gap: 24px; padding: 12px 0; margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.meta-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 13px; color: var(--text-secondary); font-weight: 500;
}
.meta-icon {
    width: 18px; height: 18px; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px;
}

/* ── Result cards ── */
.rcard {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); overflow: hidden; margin-bottom: 16px;
    transition: border-color .2s;
}
.rcard:hover { border-color: var(--border-light); }
.rcard-head {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px; background: var(--surface2);
    border-bottom: 1px solid var(--border);
}
.rcard-icon {
    width: 34px; height: 34px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.rcard-icon.s { background: var(--accent-bg); }
.rcard-icon.a { background: var(--blue-bg); }
.rcard-icon.r { background: var(--amber-bg); }
.rcard-title { font-weight: 700; font-size: 15px; color: var(--text); flex: 1; }
.rcard-count {
    font-size: 12px; font-weight: 600; color: var(--text-muted);
    background: var(--surface); padding: 3px 10px; border-radius: 100px;
}
.rcard-body { padding: 20px; }

/* ── Summary ── */
.summary-p {
    color: var(--text); font-size: 14.5px; line-height: 1.85;
    font-weight: 400;
}

/* ── Action table ── */
.atbl { width: 100%; border-collapse: collapse; }
.atbl th {
    text-align: left; font-size: 11px; font-weight: 700;
    color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.8px; padding: 10px 16px;
    border-bottom: 1px solid var(--border);
}
.atbl td {
    padding: 12px 16px; font-size: 13.5px; color: var(--text);
    border-bottom: 1px solid rgba(28,39,68,.5); vertical-align: top;
}
.atbl tr:last-child td { border: none; }
.atbl tr:hover td { background: rgba(99,91,255,.02); }
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 6px;
    font-size: 12px; font-weight: 600;
}
.pill-owner { background: var(--accent-bg); color: var(--accent-light); border: 1px solid var(--accent-border); }
.pill-dep { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber-border); }
.pill-dl { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
.dim { color: var(--text-muted); font-style: italic; font-size: 12.5px; }

/* ── Risk items ── */
.risk-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; margin-bottom: 6px;
    background: var(--amber-bg); border: 1px solid var(--amber-border);
    border-radius: var(--r-sm); font-size: 13.5px;
    line-height: 1.6; color: var(--text);
}
.risk-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--amber); flex-shrink: 0; margin-top: 7px;
}

/* ── JSON viewer ── */
.json-viewer {
    background: #070b14; border: 1px solid var(--border);
    border-radius: var(--r); padding: 20px; overflow: auto; max-height: 600px;
}
.json-viewer pre {
    margin: 0; font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12.5px; line-height: 1.7; color: var(--text);
    white-space: pre-wrap; word-break: break-word;
}
.jk { color: #7dd3fc; } .js { color: #86efac; }
.jn { color: #fbbf24; } .jl { color: #c084fc; font-style: italic; }
.jb { color: #475569; }
</style>
""", unsafe_allow_html=True)


# ── Core Functions ──

def get_model_client():
    return OpenAIChatCompletionClient(
        model=os.getenv("MODEL_NAME", "arcee-ai/trinity-large-preview:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info={
            "vision": False, "function_calling": False,
            "json_output": True, "structured_output": False, "family": "unknown",
        },
    )


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        return {"raw": text}


async def run_all_agents(chunks):
    ctx = {"entities": [], "decisions": [], "constraints": []}
    client = get_model_client()
    msg = json.dumps({"document_chunks": chunks, "global_context": ctx}, indent=2)

    team = RoundRobinGroupChat(
        participants=[
            create_summary_agent(client),
            create_action_agent(client),
            create_risk_agent(client),
        ],
        termination_condition=MaxMessageTermination(max_messages=4),
    )

    result = await team.run(task=msg)
    out = {}
    for m in result.messages:
        if m.source == "Summary_Agent":
            out["summary"] = parse_json(m.content)
        elif m.source == "Action_Agent":
            out["actions"] = parse_json(m.content)
        elif m.source == "Risk_Agent":
            out["risks"] = parse_json(m.content)
    return out


def esc(s):
    return html_lib.escape(str(s)) if s else ""


def highlight_json(obj):
    s = json.dumps(obj, indent=2)
    import re
    def rep(m):
        t = m.group(0)
        if t.startswith('"'):
            if t.endswith(':'):
                return f'<span class="jk">{esc(t[:-1])}</span>:'
            return f'<span class="js">{esc(t)}</span>'
        if t in ('true', 'false'):
            return f'<span class="js">{t}</span>'
        if t == 'null':
            return f'<span class="jl">{t}</span>'
        return f'<span class="jn">{t}</span>'
    s = re.sub(r'("(?:\\.|[^"\\])*"(?:\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d*)?)', rep, s)
    s = re.sub(r'([{}\[\]])', r'<span class="jb">\1</span>', s)
    return s


# ── UI ──

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-chip"><div class="pulse"></div><span>Multi-Agent System</span></div>
    <h1>Document Intelligence</h1>
    <p>Upload a PDF and let 3 AI agents analyze it in parallel using AutoGen RoundRobinGroupChat</p>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader("Upload a PDF document", type=["pdf"], label_visibility="collapsed")

if uploaded:
    if st.button("Analyze Document"):
        chunks = pdf_to_chunks(uploaded.read())

        status = st.empty()
        with status.container():
            for name in ["Summary Agent", "Action & Dependency Agent", "Risk & Open-Issues Agent"]:
                st.markdown(f"""
                <div class="agent-row">
                    <div class="agent-dot running"></div>
                    <div class="agent-name">{name}</div>
                    <div class="agent-tag running">Running</div>
                </div>""", unsafe_allow_html=True)

        t0 = time.time()
        results = asyncio.run(run_all_agents(chunks))
        elapsed = round(time.time() - t0, 2)

        with status.container():
            for name in ["Summary Agent", "Action & Dependency Agent", "Risk & Open-Issues Agent"]:
                st.markdown(f"""
                <div class="agent-row done">
                    <div class="agent-dot done"></div>
                    <div class="agent-name">{name}</div>
                    <div class="agent-tag done">Complete</div>
                </div>""", unsafe_allow_html=True)

        st.session_state["output"] = {
            "filename": uploaded.name,
            "chunks_processed": len(chunks),
            "processing_time_seconds": elapsed,
            "results": results,
        }

if "output" in st.session_state:
    data = st.session_state["output"]
    r = data["results"]

    # Meta
    st.markdown(f"""
    <div class="meta-bar">
        <div class="meta-item"><div class="meta-icon">📦</div>{data['chunks_processed']} chunks processed</div>
        <div class="meta-item"><div class="meta-icon">⏱️</div>{data['processing_time_seconds']}s total</div>
        <div class="meta-item"><div class="meta-icon">📄</div>{esc(data['filename'])}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Cards", "JSON"])

    with tab1:
        # Summary
        summary_text = esc(r.get("summary", {}).get("summary", str(r.get("summary", ""))))
        st.markdown(f"""
        <div class="rcard">
            <div class="rcard-head">
                <div class="rcard-icon s">📄</div>
                <div class="rcard-title">Summary</div>
            </div>
            <div class="rcard-body"><p class="summary-p">{summary_text}</p></div>
        </div>
        """, unsafe_allow_html=True)

        # Actions
        actions = r.get("actions", {}).get("actions", [])
        rows = ""
        for a in actions:
            task = esc(a.get("task", ""))
            owner = f'<span class="pill pill-owner">{esc(a["owner"])}</span>' if a.get("owner") else '<span class="dim">—</span>'
            dep = f'<span class="pill pill-dep">{esc(a["dependency"])}</span>' if a.get("dependency") else '<span class="dim">—</span>'
            dl = f'<span class="pill pill-dl">{esc(a["deadline"])}</span>' if a.get("deadline") else '<span class="dim">—</span>'
            rows += f"<tr><td>{task}</td><td>{owner}</td><td>{dep}</td><td>{dl}</td></tr>"

        st.markdown(f"""
        <div class="rcard">
            <div class="rcard-head">
                <div class="rcard-icon a">📋</div>
                <div class="rcard-title">Actions & Dependencies</div>
                <div class="rcard-count">{len(actions)} items</div>
            </div>
            <div class="rcard-body" style="padding:0;">
                <table class="atbl">
                    <thead><tr><th>Task</th><th>Owner</th><th>Dependency</th><th>Deadline</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risks
        risks = r.get("risks", {}).get("risks", [])
        items = "".join(
            f'<div class="risk-row"><div class="risk-dot"></div><div>{esc(risk)}</div></div>'
            for risk in risks
        )
        st.markdown(f"""
        <div class="rcard">
            <div class="rcard-head">
                <div class="rcard-icon r">⚡</div>
                <div class="rcard-title">Risks & Open Issues</div>
                <div class="rcard-count">{len(risks)} items</div>
            </div>
            <div class="rcard-body">{items}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown(f"""
        <div class="json-viewer"><pre>{highlight_json(data)}</pre></div>
        """, unsafe_allow_html=True)

    # Download
    st.download_button(
        label="Download Full JSON Report",
        data=json.dumps(data, indent=2),
        file_name=data["filename"].replace(".pdf", "") + "_results.json",
        mime="application/json",
        use_container_width=True,
    )
