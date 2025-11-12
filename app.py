import gradio as gr
import re
import time
import tempfile
import os
import requests
from bs4 import BeautifulSoup
from utils.research_tools import query_groq, GroqClient
from utils.logger import (
    log_interaction,
    get_history,
    clear_history,
    export_history_json,
    export_history_markdown,
)

# --- Core Functions ---

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 700

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = GroqClient()
    return _client


def _estimate_tokens(text: str) -> int:
    # very rough heuristic: 1 token ~= 4 chars
    return max(1, round(len(text) / 4))


def _apply_preset(name: str) -> str:
    presets = {
        "Standard": "",
        "Concise": "Answer concisely with bullet points when helpful.",
        "Teacher": "Explain like I'm new to the topic, with analogies and step-by-step reasoning.",
        "Developer": "Use code examples where relevant and be explicit about trade-offs.",
        "Researcher": "Provide structured analysis with assumptions, evidence, and limitations.",
    }
    return presets.get(name, "")


def _with_memory(user_query: str, use_memory: bool) -> str:
    if not use_memory:
        return user_query
    hist = get_history()[-3:]  # last 3 turns
    if not hist:
        return user_query
    lines = ["Context from earlier in this conversation:"]
    for h in hist:
        q = h.get("question", "").strip()
        a = h.get("answer", "").strip()
        if q and a:
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
    lines.append("")
    lines.append(f"Current user request: {user_query}")
    return "\n".join(lines)


def _export_markdown(text: str, prefix: str = "export") -> str:
    os.makedirs("logs", exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".md", dir="logs")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def ai_research(query, system_prompt, stream, use_memory):
    if not query.strip():
        return "Please enter a valid question or topic.", ""
    start = time.time()
    if stream:
        acc = ""
        for partial in _get_client().chat_stream(
            _with_memory(query, use_memory),
            model="llama-3.1-8b-instant",
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system_prompt or None,
        ):
            acc = partial
            elapsed = time.time() - start
            stats = f"~ tokens (prompt+completion): {_estimate_tokens(query + (system_prompt or '')) + _estimate_tokens(acc)} | time: {elapsed:.1f}s"
            yield acc, stats
        log_interaction(query, acc)
    else:
        response = query_groq(
            _with_memory(query, use_memory),
            model="llama-3.1-8b-instant",
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system_prompt or None,
        )
        log_interaction(query, response)
        elapsed = time.time() - start
        stats = f"~ tokens (prompt+completion): {_estimate_tokens(query + (system_prompt or '')) + _estimate_tokens(response)} | time: {elapsed:.1f}s"
        return response, stats

def _is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


def _fetch_url_text(url: str, max_chars: int = 6000) -> str:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text[:max_chars]
    except Exception:
        return f"[Could not fetch URL content, summarizing the URL contextually instead]\nURL: {url}"


def summarize_text_or_url(content, system_prompt, stream):
    if not content.strip():
        return "Please provide a URL or text to summarize.", ""
    source_text = _fetch_url_text(content) if _is_url(content) else content
    prompt = (
        "You are a world-class summarizer. Produce a concise, faithful summary with: "
        "- title\n- key points\n- important quotes\n- a short TL;DR.\n\n"
        f"Content to summarize (may be HTML-extracted):\n\n{source_text}"
    )
    start = time.time()
    if stream:
        acc = ""
        for partial in _get_client().chat_stream(
            prompt,
            model="llama-3.1-8b-instant",
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system_prompt or "Summarize clearly in Markdown.",
        ):
            acc = partial
            elapsed = time.time() - start
            stats = f"~ tokens (prompt+completion): {_estimate_tokens(prompt + (system_prompt or '')) + _estimate_tokens(acc)} | time: {elapsed:.1f}s"
            yield acc, stats
        log_interaction(content[:100] + "...", acc)
    else:
        response = query_groq(
            prompt,
            model="llama-3.1-8b-instant",
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system_prompt or "Summarize clearly in Markdown.",
        )
        log_interaction(content[:100] + "...", response)
        elapsed = time.time() - start
        stats = f"~ tokens (prompt+completion): {_estimate_tokens(prompt + (system_prompt or '')) + _estimate_tokens(response)} | time: {elapsed:.1f}s"
        return response, stats

def view_history():
    history = get_history()
    if not history:
        return "No history available yet."
    formatted = ""
    for h in reversed(history):
        formatted += f"### {h['timestamp']}\n**Q:** {h['question']}\n\n**A:** {h['answer']}\n\n---\n"
    return formatted


# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .main-nav button:hover { background-color: #2f3136 !important; color: white !important; }
    .gr-button-primary { border-radius: 8px !important; }
    .gradio-container { font-family: 'Inter', sans-serif; }
    .tab-title { font-weight: 600; font-size: 1.2em; margin-bottom: 10px; }
""") as demo:
    gr.Markdown("# 🧠 GroqAI Research Companion\nEmpowered by LLaMA 3 • Built for researchers, developers, and creators.")
    
    with gr.Row(elem_classes="main-nav"):
        with gr.Column():
            with gr.Tab("🔍 Research Assistant"):
                gr.Markdown("<div class='tab-title'>Ask anything — get structured, AI-powered responses</div>")
                with gr.Row():
                    stream_cb = gr.Checkbox(True, label="Stream output")
                    memory_cb = gr.Checkbox(True, label="Use memory (last 3 turns)")
                with gr.Row():
                    preset_dd = gr.Dropdown(
                        label="Style preset",
                        choices=["Standard", "Concise", "Teacher", "Developer", "Researcher"],
                        value="Standard",
                    )
                system_tb = gr.Textbox(label="System instruction (optional)", placeholder="e.g., Answer concisely with examples where helpful.")
                query_input = gr.Textbox(label="Enter your question", placeholder="e.g. Explain quantum computing in simple terms")
                query_button = gr.Button("Run Research")
                query_output = gr.Markdown()
                query_stats = gr.Markdown(label="Stats")
                export_research_btn = gr.Button("Export Markdown")
                export_research_msg = gr.Markdown(label="Export status")
                preset_dd.change(_apply_preset, inputs=preset_dd, outputs=system_tb)
                query_button.click(
                    ai_research,
                    inputs=[
                        query_input,
                        system_tb,
                        stream_cb,
                        memory_cb,
                    ],
                    outputs=[query_output, query_stats],
                )
                export_research_btn.click(lambda t: f"Saved to: {_export_markdown(t, 'research')}", inputs=query_output, outputs=export_research_msg)

            with gr.Tab("📰 Summarize URL / Text"):
                gr.Markdown("<div class='tab-title'>Paste a URL or long text for summarization</div>")
                text_input = gr.Textbox(label="Enter text or URL", lines=5, placeholder="Paste text or a webpage link here")
                with gr.Row():
                    sum_stream_cb = gr.Checkbox(True, label="Stream output")
                sum_system_tb = gr.Textbox(label="Summarizer instruction (optional)", value="Summarize clearly in Markdown.")
                summarize_button = gr.Button("Summarize")
                summary_output = gr.Markdown()
                summary_stats = gr.Markdown(label="Stats")
                export_summary_btn = gr.Button("Export Markdown")
                export_summary_msg = gr.Markdown(label="Export status")
                summarize_button.click(
                    summarize_text_or_url,
                    inputs=[
                        text_input,
                        sum_system_tb,
                        sum_stream_cb,
                    ],
                    outputs=[summary_output, summary_stats],
                )
                export_summary_btn.click(lambda t: f"Saved to: {_export_markdown(t, 'summary')}", inputs=summary_output, outputs=export_summary_msg)

            with gr.Tab("🕓 History"):
                gr.Markdown("<div class='tab-title'>View your past research interactions</div>")
                with gr.Row():
                    view_button = gr.Button("Load History")
                    clear_button = gr.Button("Clear History")
                    export_json_btn = gr.Button("Export JSON")
                    export_md_btn = gr.Button("Export Markdown")
                history_output = gr.Markdown()
                export_output = gr.Textbox(label="Exported Content", lines=10)
                view_button.click(view_history, outputs=history_output)
                clear_button.click(lambda: (clear_history() or "History cleared."), outputs=history_output)
                export_json_btn.click(export_history_json, outputs=export_output)
                export_md_btn.click(export_history_markdown, outputs=export_output)

demo.launch(share=True)
