import os
import json
from utils.logger import log_interaction, get_history, clear_history, export_history_json, export_history_markdown


def setup_function(function):
    clear_history()


def test_log_and_get_history():
    log_interaction("Q1", "A1")
    log_interaction("Q2", "A2")
    hist = get_history()
    assert len(hist) == 2
    assert hist[0]["question"] == "Q1"
    assert hist[1]["answer"] == "A2"


def test_export_json_and_markdown():
    log_interaction("Why?", "Because.")
    js = export_history_json()
    data = json.loads(js)
    assert data[0]["question"] == "Why?"
    md = export_history_markdown()
    assert "Because." in md

