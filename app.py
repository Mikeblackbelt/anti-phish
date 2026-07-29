#!/usr/bin/env python3
"""
AntiPhish — Modern UI that ties together:
  - Trained XGBoost model (phishing_model.joblib from train.py)
  - Free Groq LLM detector
  - Live IMAP inbox scanning (Gmail / Outlook / any IMAP)

Run:
  pip install streamlit pandas numpy joblib xgboost python-dotenv
  export GROQ_API_KEY=gsk_...
  streamlit run app.py
"""

from __future__ import annotations

import email
import imaplib
import json
import os
import random
import re
import ssl
from email.header import decode_header
from html import unescape
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import urllib.error
import urllib.request

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_FEATURE_COLS = [
    "num_words",
    "num_unique_words",
    "num_stopwords",
    "num_links",
    "num_unique_domains",
    "num_email_addresses",
    "num_spelling_errors",
    "num_urgent_keywords",
]

STOPWORDS = {
    "the", "and", "a", "to", "of", "in", "is", "for", "on", "that", "by",
    "this", "with", "i", "you", "it", "not", "or", "be", "are", "from",
    "at", "as", "your", "have", "was", "but", "we", "an", "they", "which",
    "will", "all", "can", "if", "do", "about", "my", "so", "has", "been",
}

URGENT_KEYWORDS = {
    "urgent", "immediately", "asap", "verify", "verification", "confirm",
    "confirmation", "update", "password", "account", "suspended", "locked",
    "unusual", "activity", "security", "alert", "warning", "expire",
    "expired", "click", "login", "sign-in", "signin", "credential",
    "unauthorized", "restricted", "action required", "act now", "final notice",
    "limited time", "suspend", "deactivate", "locked out",
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Pool of free Groq models — one is chosen at random per request to
# reduce the chance of hitting a single model's rate limit.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]
DEFAULT_MODEL = GROQ_MODELS[0]

LLM_SYSTEM = """You are a cybersecurity analyst specializing in email phishing detection.
Analyze the email text and decide whether it is phishing or legitimate.

Respond with ONLY a single JSON object (no markdown fences) using exactly these keys:
{
  "label": "phishing" or "legitimate",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<short explanation>",
  "red_flags": ["..."],
  "safe_signals": ["..."]
}
Be strict on urgency, credential requests, suspicious links, and impersonation.
"""

IMAP_PRESETS = {
    "Gmail": ("imap.gmail.com", 993),
    "Outlook / Hotmail": ("outlook.office365.com", 993),
    "Yahoo": ("imap.mail.yahoo.com", 993),
    "Custom": ("", 993),
}


# ---------------------------------------------------------------------------
# Feature extraction (mirrors train.py engineering from raw text)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text, flags=re.I)


def _domain(url: str) -> str:
    url = re.sub(r"^https?://", "", url, flags=re.I)
    url = re.sub(r"^www\.", "", url, flags=re.I)
    return url.split("/")[0].split("?")[0].lower()


def extract_raw_features(text: str) -> dict[str, float]:
    """Approximate the 8 Kaggle engineered features from plain email text."""
    text = text or ""
    tokens = _tokenize(text)
    words = tokens
    unique = set(words)
    stop_count = sum(1 for w in words if w in STOPWORDS)
    urls = _extract_urls(text)
    domains = {_domain(u) for u in urls if _domain(u)}
    email_addrs = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)

    # Lightweight "spelling error" proxy: tokens with digits mixed in or repeated chars
    spell_err = 0
    for w in words:
        if len(w) >= 4 and re.search(r"(.)\1{2,}", w):
            spell_err += 1
        elif re.search(r"[a-zA-Z]+\d+[a-zA-Z]*|\d+[a-zA-Z]+", w) and "@" not in w:
            spell_err += 1

    joined = " ".join(words)
    urgent = sum(1 for kw in URGENT_KEYWORDS if kw in joined)

    return {
        "num_words": float(len(words)),
        "num_unique_words": float(len(unique)),
        "num_stopwords": float(stop_count),
        "num_links": float(len(urls)),
        "num_unique_domains": float(len(domains)),
        "num_email_addresses": float(len(email_addrs)),
        "num_spelling_errors": float(spell_err),
        "num_urgent_keywords": float(urgent),
    }


def engineer_features_row(raw: dict[str, float]) -> dict[str, float]:
    """Same ratio / interaction features as train.py."""
    out = dict(raw)
    words = max(out["num_words"], 1.0)
    links = out["num_links"]
    links_safe = links if links > 0 else np.nan

    out["unique_word_ratio"] = out["num_unique_words"] / words
    out["stopword_ratio"] = out["num_stopwords"] / words
    out["spelling_error_rate"] = out["num_spelling_errors"] / words
    out["urgent_keyword_density"] = out["num_urgent_keywords"] / words
    out["links_per_word"] = out["num_links"] / words
    out["email_addrs_per_word"] = out["num_email_addresses"] / words
    out["domains_per_link"] = (out["num_unique_domains"] / links_safe) if links > 0 else 0.0
    out["stopword_to_unique"] = out["num_stopwords"] / max(out["num_unique_words"], 1.0)
    out["error_to_unique"] = out["num_spelling_errors"] / max(out["num_unique_words"], 1.0)

    out["has_links"] = 1.0 if out["num_links"] > 0 else 0.0
    out["has_multiple_links"] = 1.0 if out["num_links"] >= 2 else 0.0
    out["has_multiple_domains"] = 1.0 if out["num_unique_domains"] >= 2 else 0.0
    out["has_email_address"] = 1.0 if out["num_email_addresses"] > 0 else 0.0
    out["has_spelling_errors"] = 1.0 if out["num_spelling_errors"] > 0 else 0.0
    out["has_urgent_keywords"] = 1.0 if out["num_urgent_keywords"] > 0 else 0.0
    out["high_urgency"] = 1.0 if out["num_urgent_keywords"] >= 2 else 0.0
    out["is_short"] = 1.0 if out["num_words"] <= 50 else 0.0
    out["is_very_short"] = 1.0 if out["num_words"] <= 20 else 0.0
    out["is_long"] = 1.0 if out["num_words"] >= 300 else 0.0
    out["urgent_and_has_links"] = out["has_urgent_keywords"] * out["has_links"]
    out["urgent_links_score"] = out["num_urgent_keywords"] * out["num_links"]
    out["errors_and_links"] = out["has_spelling_errors"] * out["has_links"]

    for col in RAW_FEATURE_COLS:
        out[f"log1p_{col}"] = float(np.log1p(max(out[col], 0.0)))

    return out


# ---------------------------------------------------------------------------
# Model + LLM
# ---------------------------------------------------------------------------

@st.cache_resource
def load_xgb_model(path: str):
    bundle = joblib.load(path)
    # Support both old ("pipeline") and new ("model") formats
    if "model" in bundle:
        return bundle["model"], bundle.get("feature_cols", []), bundle.get("decision_threshold", 0.5)
    if "pipeline" in bundle:
        return bundle["pipeline"], bundle.get("feature_cols", []), bundle.get("decision_threshold", 0.5)
    raise ValueError("Unrecognized model bundle keys")


def predict_xgb(model, feature_cols: list[str], threshold: float, text: str) -> dict[str, Any]:
    raw = extract_raw_features(text)
    feats = engineer_features_row(raw)
    if feature_cols:
        row = {c: feats.get(c, 0.0) for c in feature_cols}
        X = pd.DataFrame([row])[feature_cols]
    else:
        X = pd.DataFrame([feats])
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception:
        # Some pipelines expect different column order / types
        proba = float(model.predict_proba(X.values)[0, 1])
    label = "phishing" if proba >= threshold else "legitimate"
    return {
        "label": label,
        "probability": proba,
        "threshold": threshold,
        "raw_features": raw,
    }


def call_groq(email_text: str, api_key: str, model: str | None = None) -> dict:
    if model is None:
        model = random.choice(GROQ_MODELS)
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": f"Analyze this email for phishing:\n\n---\n{email_text.strip()[:8000]}\n---"},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AntiPhish/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    result = json.loads(content)
    result["_model_used"] = model  # diagnostic: which model answered
    return result


def combined_risk(xgb: dict | None, llm: dict | None) -> tuple[str, str, float]:
    """Merge signals into overall risk: High / Medium / Low + score 0-100."""
    score = 0.0
    weights = 0.0
    if xgb:
        score += xgb["probability"] * 45
        weights += 45
    if llm:
        conf = float(llm.get("confidence") or 0.5)
        is_phish = str(llm.get("label", "")).lower() == "phishing"
        score += (conf if is_phish else (1 - conf)) * 55
        weights += 55
    if weights == 0:
        return "Unknown", "gray", 0.0
    score = score / weights * 100
    if score >= 65:
        return "High", "red", score
    if score >= 35:
        return "Medium", "orange", score
    return "Low", "green", score


# ---------------------------------------------------------------------------
# IMAP helpers
# ---------------------------------------------------------------------------

def _decode_mime(s) -> str:
    if s is None:
        return ""
    parts = decode_header(s)
    out = []
    for frag, enc in parts:
        if isinstance(frag, bytes):
            out.append(frag.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(str(frag))
    return " ".join(out)


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", unescape(html)).strip()


def extract_body(msg: email.message.Message) -> str:
    texts, htmls = [], []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain":
                texts.append(decoded)
            elif ctype == "text/html":
                htmls.append(decoded)
    else:
        try:
            payload = msg.get_payload(decode=True) or b""
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                htmls.append(decoded)
            else:
                texts.append(decoded)
        except Exception:
            pass
    if texts:
        return "\n".join(texts).strip()
    if htmls:
        return _html_to_text("\n".join(htmls))
    return ""


def fetch_emails(
    host: str,
    port: int,
    username: str,
    password: str,
    mailbox: str = "INBOX",
    limit: int = 20,
) -> list[dict]:
    ctx = ssl.create_default_context()
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=ctx)
    mail.login(username, password)
    mail.select(mailbox, readonly=True)
    status, data = mail.search(None, "ALL")
    if status != "OK":
        mail.logout()
        raise RuntimeError("IMAP search failed")
    ids = data[0].split()
    ids = ids[-limit:]  # most recent
    results = []
    for eid in reversed(ids):
        status, msg_data = mail.fetch(eid, "(RFC822)")
        if status != "OK":
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        subject = _decode_mime(msg.get("Subject"))
        sender = _decode_mime(msg.get("From"))
        date = msg.get("Date", "")
        body = extract_body(msg)
        results.append({
            "id": eid.decode() if isinstance(eid, bytes) else str(eid),
            "subject": subject,
            "from": sender,
            "date": date,
            "body": body,
            "preview": (body[:200] + "…") if len(body) > 200 else body,
        })
    mail.logout()
    return results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def risk_badge(level: str, color: str, score: float) -> None:
    st.markdown(
        f"""
        <div style="
            display:inline-block;padding:6px 14px;border-radius:999px;
            background:{color};color:white;font-weight:700;font-size:0.95rem;
            letter-spacing:0.03em;">
            {level} RISK · {score:.0f}/100
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(xgb_res: dict | None, llm_res: dict | None) -> None:
    level, color, score = combined_risk(xgb_res, llm_res)
    risk_badge(level, color, score)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("XGBoost model")
        if xgb_res:
            st.metric("Label", xgb_res["label"].upper())
            st.metric("Phishing probability", f"{xgb_res['probability']:.3f}")
            st.caption(f"Decision threshold: {xgb_res['threshold']:.3f}")
            with st.expander("Extracted features"):
                st.json(xgb_res["raw_features"])
        else:
            st.info("Model not loaded or skipped.")

    with c2:
        st.subheader("LLM analyst")
        if llm_res:
            st.metric("Label", str(llm_res.get("label", "?")).upper())
            st.metric("Confidence", f"{float(llm_res.get('confidence') or 0):.2f}")
            model_used = llm_res.get("_model_used")
            if model_used:
                st.caption(f"Model: `{model_used}`")
            st.write(llm_res.get("reasoning", ""))
            red = llm_res.get("red_flags") or []
            safe = llm_res.get("safe_signals") or []
            if red:
                st.markdown("**Red flags**")
                for r in red:
                    st.markdown(f"- 🚩 {r}")
            if safe:
                st.markdown("**Safe signals**")
                for s in safe:
                    st.markdown(f"- ✅ {s}")
        else:
            st.info("LLM not run (missing API key or skipped).")


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AntiPhish",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛡️ AntiPhish")
    st.caption("Scan your inbox or paste an email · XGBoost + free Groq LLM")

    # ----- Sidebar config -----
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input(
            "Model path",
            value=os.environ.get("PHISH_MODEL", "phishing_model.joblib"),
        )
        groq_key = st.text_input(
            "Groq API key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help="Free key: https://console.groq.com/keys",
        )
        use_xgb = st.toggle("Use XGBoost model", value=True)
        use_llm = st.toggle("Use LLM detector", value=True)
        st.divider()
        st.markdown("**IMAP presets**")
        preset = st.selectbox("Provider", list(IMAP_PRESETS.keys()))
        default_host, default_port = IMAP_PRESETS[preset]
        imap_host = st.text_input("IMAP host", value=default_host)
        imap_port = st.number_input("Port", value=default_port, step=1)
        st.caption("Gmail: enable 2FA → create an [App Password](https://myaccount.google.com/apppasswords).")

    # Load model once
    model = feature_cols = threshold = None
    if use_xgb:
        if os.path.isfile(model_path):
            try:
                model, feature_cols, threshold = load_xgb_model(model_path)
                st.sidebar.success(f"Model loaded · threshold={threshold:.3f}")
            except Exception as e:
                st.sidebar.error(f"Model load failed: {e}")
                use_xgb = False
        else:
            st.sidebar.warning(f"Model file not found: {model_path}")
            use_xgb = False

    tab_inbox, tab_paste, tab_about = st.tabs(["📬 Inbox scan", "✍️ Paste email", "ℹ️ About"])

    # ----- Inbox tab -----
    with tab_inbox:
        st.subheader("Connect to your mailbox")
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            imap_user = st.text_input("Email address", key="imap_user")
        with col_b:
            imap_pass = st.text_input("Password / App password", type="password", key="imap_pass")
        with col_c:
            limit = st.number_input("Max emails", min_value=1, max_value=50, value=15)

        if st.button("Fetch & scan inbox", type="primary", use_container_width=True):
            if not imap_host or not imap_user or not imap_pass:
                st.error("Host, email, and password are required.")
            else:
                with st.spinner("Connecting via IMAP and downloading messages…"):
                    try:
                        emails = fetch_emails(
                            imap_host, int(imap_port), imap_user, imap_pass, limit=int(limit)
                        )
                        st.session_state["inbox_emails"] = emails
                        st.success(f"Fetched {len(emails)} messages.")
                    except imaplib.IMAP4.error as e:
                        st.error(f"IMAP auth/error: {e}")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        emails = st.session_state.get("inbox_emails") or []
        if emails:
            st.divider()
            st.subheader(f"Results ({len(emails)} emails)")
            for i, em in enumerate(emails):
                text = f"Subject: {em['subject']}\nFrom: {em['from']}\n\n{em['body']}"
                with st.expander(f"{em['subject'] or '(no subject)'}  —  {em['from']}", expanded=(i == 0)):
                    st.caption(em.get("date", ""))
                    st.text(em["preview"] or "(empty body)")

                    xgb_res = llm_res = None
                    err_box = st.empty()
                    try:
                        if use_xgb and model is not None:
                            xgb_res = predict_xgb(model, feature_cols, threshold, text)
                        if use_llm and groq_key:
                            llm_res = call_groq(text, groq_key)
                        elif use_llm and not groq_key:
                            err_box.warning("Set Groq API key in the sidebar to enable LLM analysis.")
                    except Exception as e:
                        err_box.error(str(e))

                    render_result(xgb_res, llm_res)

    # ----- Paste tab -----
    with tab_paste:
        st.subheader("Analyze a single email")
        subject = st.text_input("Subject (optional)")
        sender = st.text_input("From (optional)")
        body = st.text_area("Email body", height=220, placeholder="Paste the full email text here…")
        if st.button("Analyze", type="primary", use_container_width=True):
            if not body.strip():
                st.warning("Paste some email text first.")
            else:
                text = ""
                if subject:
                    text += f"Subject: {subject}\n"
                if sender:
                    text += f"From: {sender}\n"
                text += f"\n{body}"
                xgb_res = llm_res = None
                with st.spinner("Running detectors…"):
                    try:
                        if use_xgb and model is not None:
                            xgb_res = predict_xgb(model, feature_cols, threshold, text)
                        if use_llm and groq_key:
                            llm_res = call_groq(text, groq_key)
                        elif use_llm and not groq_key:
                            st.warning("Set Groq API key in the sidebar for LLM analysis.")
                    except Exception as e:
                        st.error(str(e))
                render_result(xgb_res, llm_res)

    # ----- About -----
    with tab_about:
        st.markdown(
            """
### How AntiPhish works

| Component | Role |
|---|---|
| **XGBoost model** | Your trained `phishing_model.joblib` — uses engineered numeric features extracted from the email text (word counts, links, urgency keywords, etc.). |
| **Groq LLM** | Free Llama / Gemma / Mixtral models (randomly rotated per request to avoid rate limits) that read the raw email and return label, confidence, red flags. |
| **Combined risk** | Weighted blend (model ~45%, LLM ~55%) → High / Medium / Low. |

### Inbox access
Uses standard **IMAP** (read-only). For Gmail you need an [App Password](https://myaccount.google.com/apppasswords) (2FA required). Credentials are used only for the live session and are not stored.

### Files this UI wires together
- `train.py` → produces `phishing_model.joblib`
- `predict_from_trained.py` → tabular inference logic (reimplemented here with feature extraction)
- `detect_with_llm.py` → Groq phishing prompt

### Privacy
Emails are sent to Groq only if the LLM toggle is on. The XGBoost path runs fully locally.
            """
        )


if __name__ == "__main__":
    main()