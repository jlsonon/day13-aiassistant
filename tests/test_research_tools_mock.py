import json
from unittest.mock import patch, MagicMock
from utils.research_tools import GroqClient


def test_client_formats_payload_and_parses_response():
    client = GroqClient(api_key="test_key")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "choices": [
            {"message": {"content": "Hello world"}}
        ]
    }
    with patch.object(client.session, "post", return_value=fake_resp) as post:
        out = client.chat("Hi", model="llama3-70b-8192", max_tokens=16)
    assert out == "Hello world"
    # Check payload contains required fields
    args, kwargs = post.call_args
    payload = json.loads(kwargs["data"])  # we send serialized JSON
    assert payload["model"] == "llama3-70b-8192"
    assert payload["max_tokens"] == 16
    assert payload["messages"][0]["role"] in ("system", "user")


def test_stream_yields_accumulated():
    client = GroqClient(api_key="test_key")
    # Simulate SSE data chunks
    stream_lines = [
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}",
        b"data: [DONE]",
    ]
    fake_resp = MagicMock()
    fake_resp.iter_lines.return_value = (l.decode("utf-8") for l in stream_lines)
    with patch.object(client, "_post", return_value=fake_resp):
        chunks = list(client.chat_stream("Hi", max_tokens=16))
    assert chunks[-1] == "Hello"
