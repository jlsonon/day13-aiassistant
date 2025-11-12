# 🧠 GroqAI Research Companion

An elegant Gradio app powered by Groq's blazing-fast LLMs. Ask research questions, summarize articles or pasted text, and manage your interaction history — with streaming, model controls, and portfolio-ready UX.

## Features

- Chat-style research assistant with structured, high-quality responses
- Streaming output for fast feedback
- Model selection, temperature and max-tokens controls
- Optional system prompts to steer behavior
- Summarize URL or raw text (auto-fetches web page content and cleans it)
- History viewer, clear, and one-click export (JSON/Markdown)
- Robust Groq client:
  - Retries with exponential backoff on transient errors
  - Clear error messages for 400/401/429/5xx
  - Basic rate limiting to avoid spamming

## Tech Stack

- Python, Gradio UI
- Groq Chat Completions API
- Requests, BeautifulSoup for URL text extraction

## Quick Start

Step 1 — Create a virtual environment (optional but recommended)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

Step 3 — Configure environment variables

Create a `.env` file in the project root with:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Step 4 — Run the app

```powershell
python app.py
```

Gradio will print a local URL you can open in your browser.

## Environment

- Required: `GROQ_API_KEY` (starts with `gsk_...`)

## Troubleshooting

- 400 invalid_request_error:
  - Ensure a valid model (default: `llama-3.1-8b-instant`)
  - Set `max_tokens > 0`
  - Verify the request `messages` structure
- 401 authentication_error:
  - Verify your GROQ_API_KEY is present and correct
- Timeouts / 5xx / 429:
  - The client retries automatically with backoff; try again later if the issue persists

## File Structure

- `app.py` — Gradio UI and UI logic
- `utils/research_tools.py` — Groq client (streaming, retries, error handling)
- `utils/logger.py` — Local history store and export helpers
- `logs/history.json` — Saved Q/A history
- `requirements.txt` — Dependencies

## Notes

- The app avoids logging secrets and only stores Q/A content locally.
- For production, consider adding persistent storage (SQLite) and auth.

## License

MIT

