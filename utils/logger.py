import json
import os
from datetime import datetime

LOG_PATH = "logs/history.json"

def log_interaction(question, answer):
    os.makedirs("logs", exist_ok=True)
    try:
        with open(LOG_PATH, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    })

    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)

def get_history():
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def clear_history():
    """Delete all saved history."""
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump([], f)


def export_history_json() -> str:
    """Return history as a JSON string."""
    return json.dumps(get_history(), indent=2)


def export_history_markdown() -> str:
    """Return history as a Markdown string for easy sharing."""
    history = get_history()
    if not history:
        return "No history available."
    lines = ["# Research History\n"]
    for h in reversed(history):
        lines.append(f"## {h['timestamp']}")
        lines.append(f"**Q:** {h['question']}")
        lines.append("")
        lines.append(f"**A:**\n\n{h['answer']}")
        lines.append("\n---\n")
    return "\n".join(lines)
