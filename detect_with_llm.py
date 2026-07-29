"""
llm_phishing_detect.py — Detect phishing emails with a free LLM API.

Uses Groq's free tier (OpenAI-compatible). Get a key at:
  https://console.groq.com/keys

Usage:
  export GROQ_API_KEY="gsk_..."
  python llm_phishing_detect.py "Your account will be locked. Click here: http://bit.ly/evil"
  python llm_phishing_detect.py --file email.txt
  echo "body text" | python llm_phishing_detect.py --stdin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import dotenv

dotenv.load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are a cybersecurity analyst specializing in email phishing detection.
Analyze the email text the user provides and decide whether it is phishing or legitimate.

Respond with ONLY a single JSON object (no markdown fences, no extra text) using exactly these keys:
{
  "label": "phishing" or "legitimate",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<short explanation of the main signals>",
  "red_flags": ["list", "of", "suspicious", "indicators"],
  "safe_signals": ["list", "of", "signals", "that", "look", "benign"]
}

Be strict: urgent language, credential requests, suspicious links, sender impersonation,
poor grammar combined with action pressure, and mismatched domains are strong phishing signals.
"""


def call_groq(email_text: str, api_key: str, model: str) -> dict:
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Analyze this email for phishing:\n\n---\n{email_text.strip()}\n---",
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "llm-phishing-detect/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Groq API error {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Network error calling Groq: {e}") from e

    try:
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise SystemExit(f"Unexpected API response shape: {body}") from e


def read_input(args: argparse.Namespace) -> str:
    if args.stdin:
        return sys.stdin.read()
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read()
    if args.text:
        return " ".join(args.text)
    raise SystemExit("Provide email text as arguments, --file, or --stdin.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phishing detection via free Groq LLM API")
    parser.add_argument("text", nargs="*", help="Email body / subject text")
    parser.add_argument("--file", "-f", help="Read email text from a file")
    parser.add_argument("--stdin", action="store_true", help="Read email text from stdin")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw JSON only")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing API key. Get a free one at https://console.groq.com/keys\n"
            "then:  export GROQ_API_KEY='gsk_...'"
        )

    email_text = read_input(args)
    if not email_text.strip():
        raise SystemExit("Email text is empty.")

    result = call_groq(email_text, args.api_key, args.model)

    if args.raw:
        print(json.dumps(result, indent=2))
        return

    label = str(result.get("label", "unknown")).upper()
    conf = result.get("confidence", "?")
    print(f"Label:      {label}")
    print(f"Confidence: {conf}")
    print(f"Reasoning:  {result.get('reasoning', '')}")
    red = result.get("red_flags") or []
    safe = result.get("safe_signals") or []
    if red:
        print("Red flags:")
        for item in red:
            print(f"  - {item}")
    if safe:
        print("Safe signals:")
        for item in safe:
            print(f"  - {item}")


if __name__ == "__main__":
    main()