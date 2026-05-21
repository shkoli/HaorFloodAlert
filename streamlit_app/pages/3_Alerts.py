"""
3_Alerts.py — SMS + Email Flood Alert System
HaorFloodAlert — BulkSMSBD + Gmail SMTP community alerts for Sunamganj Haor.

Tabs:
  Manual    — test single SMS/email; Send All to contacts; WhatsApp copy text
  Automated — toggle auto-send when flood probability exceeds threshold
"""

import csv
import io
import json
import smtplib
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).parent.parent.parent
_SETTINGS_PATH = _ROOT / "alert_settings.json"

st.set_page_config(page_title="Alerts", page_icon="🚨", layout="wide")
st.title("🚨 Flood Alert System")
st.subheader("সুনামগঞ্জ হাওর — SMS + Email Community Alerts")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def barak_text(discharge: float) -> str:
    """Bangla Barak river status line for SMS."""
    if discharge >= 7500:
        return "অনেক পানি আসতেছে!!"
    if discharge >= 6000:
        return "ভারত থেকে পানি আসতেছে!"
    if discharge >= 4000:
        return "পানি বাড়তেছে"
    return "পানি ঠিক আছে"


def build_sms(risk: str, barak_discharge: float, prob: float) -> str:
    """Farmer-friendly Bengali SMS — content varies by risk level and Boro season."""
    month   = datetime.now().month
    is_boro = month in [1, 2, 3, 4, 5]
    barak   = barak_text(barak_discharge)

    if risk in ("EXTREME", "HIGH"):
        if is_boro:
            return (
                "🔴বন্যা সতর্কতা-সুনামগঞ্জ\n"
                "বন্যা আসছে! সাবধান!\n"
                f"{barak}\n"
                "ধান কাটতে পারলে কেটে ফেলুন!\n"
                "গরু-ছাগল উঁচুতে নিন\n"
                "-HaorFloodAlert"
            )
        return (
            "🔴বন্যা সতর্কতা-সুনামগঞ্জ\n"
            "বন্যা আসছে! সাবধান!\n"
            f"{barak}\n"
            "গরু-ছাগল উঁচুতে নিন\n"
            "ধান-চাল সরিয়ে রাখুন\n"
            "-HaorFloodAlert"
        )

    if risk == "MEDIUM":
        return (
            "⚠️সুনামগঞ্জ বন্যা খবর\n"
            "বন্যার সম্ভাবনা আছে\n"
            f"{barak}\n"
            "সতর্ক থাকুন\n"
            "-HaorFloodAlert"
        )

    # LOW
    return (
        "✅সুনামগঞ্জ\n"
        "বন্যার ভয় নাই\n"
        "-HaorFloodAlert"
    )


def prob_to_risk(prob: float) -> str:
    if prob >= 85: return "EXTREME"
    if prob >= 65: return "HIGH"
    if prob >= 40: return "MEDIUM"
    return "LOW"


def risk_badge(risk: str) -> tuple:
    """Returns (label_bn, hex_color)."""
    return {
        "EXTREME": ("🔴 অত্যন্ত বিপদজনক", "#CC0000"),
        "HIGH":    ("🟠 উচ্চ ঝুঁকি",       "#E65C00"),
        "MEDIUM":  ("🟡 মাঝারি ঝুঁকি",     "#B8860B"),
        "LOW":     ("🟢 স্বাভাবিক",         "#1A7A4A"),
    }[risk]


def build_email_body(risk: str, prob: float,
                     barak_discharge: float, sms_text: str) -> tuple:
    """Returns (subject, plain-text body)."""
    label, _ = risk_badge(risk)
    subject  = f"[HaorFloodAlert] {label} — সুনামগঞ্জ বন্যা সতর্কতা"
    month    = datetime.now().month
    is_boro  = month in [1, 2, 3, 4, 5]
    season   = "🌾 বোরো মৌসুম চলছে — ধান ঝুঁকিতে আছে।" if is_boro else "বোরো মৌসুম নয়।"
    body = (
        f"HaorFloodAlert — সুনামগঞ্জ হাওর বন্যা পূর্বাভাস\n"
        f"{'='*52}\n\n"
        f"ঝুঁকির মাত্রা  : {label}\n"
        f"বন্যার সম্ভাবনা: {prob:.0f}%\n"
        f"বরাক নদী       : {barak_text(barak_discharge)}  ({barak_discharge:,.0f} m³/s)\n"
        f"{season}\n\n"
        f"SMS বার্তা:\n{sms_text}\n\n"
        f"{'='*52}\n"
        f"⚠️  এটি একটি স্বয়ংক্রিয় সতর্কতা।\n"
        f"সর্বদা FFWC (ffwc.gov.bd) তথ্য দিয়ে যাচাই করুন।\n\n"
        f"HaorFloodAlert v2.0 | RTM Al-Kabir Technical University\n"
        f"প্রেরণের সময়: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    return subject, body


# ── SMS via BulkSMSBD ─────────────────────────────────────────────────────────
def send_sms(number: str, message: str,
             api_key: str, sender_id: str) -> tuple:
    """POST to bulksmsbd.net/api/smsapi. Returns (ok, detail_str)."""
    try:
        # Normalise to 880xxxxxxxx
        number = "".join(c for c in number if c.isdigit())
        if number.startswith("0"):
            number = "880" + number[1:]
        elif not number.startswith("880"):
            number = "880" + number

        r = requests.post(
            "http://bulksmsbd.net/api/smsapi",
            data={
                "api_key":  api_key,
                "type":     "text",
                "number":   number,
                "senderid": sender_id,
                "message":  message,
            },
            timeout=15,
        )
        try:
            result = r.json()
        except Exception:
            result = {"response_code": r.status_code, "raw": r.text[:120]}

        # BulkSMSBD: response_code 202 = success
        ok = str(result.get("response_code", "")) == "202"
        return ok, json.dumps(result)
    except Exception as exc:
        return False, str(exc)


# ── Email via Gmail SMTP SSL ──────────────────────────────────────────────────
def send_email(to: str, subject: str, body: str,
               sender: str, password: str) -> tuple:
    """SMTP_SSL port 465. Returns (ok, detail_str)."""
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = to
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, [to], msg.as_string())
        return True, "sent"
    except Exception as exc:
        return False, str(exc)


# ── Settings ──────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"api_key": "", "sender_id": "", "gmail": "",
            "password": "", "contacts": []}


def save_settings(d: dict) -> None:
    _SETTINGS_PATH.write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Session state init ────────────────────────────────────────────────────────
if "alert_history" not in st.session_state:
    st.session_state["alert_history"] = []
if "auto_on" not in st.session_state:
    st.session_state["auto_on"] = False
if "auto_last_sent" not in st.session_state:
    st.session_state["auto_last_sent"] = None


def append_history(channel: str, recipient: str,
                   risk: str, ok: bool, detail: str = "") -> None:
    st.session_state["alert_history"].append({
        "Time":      datetime.now().strftime("%H:%M:%S"),
        "Channel":   channel,
        "Recipient": recipient[:35],
        "Risk":      risk,
        "Status":    "✅ Sent" if ok else "❌ Failed",
        "Detail":    detail[:70],
    })


# ── Sample CSV ────────────────────────────────────────────────────────────────
# phone cells use ="..." so Excel opens them as text (preserves leading zero).
# utf-8-sig BOM is added at encode time so Excel auto-detects UTF-8.
_SAMPLE_CSV = (
    "name,phone,email\n"
    "Rahim Uddin (Farmer),=\"01712345678\",rahim@gmail.com\n"
    "UP Chairman Karim,=\"01823456789\",\n"
    "DDMC Focal Point,,ddmc.sunamganj@gov.bd\n"
    "Fishermen Association,=\"01934567890\",fishermen@example.com\n"
)

# ── Load saved settings ───────────────────────────────────────────────────────
cfg = load_settings()


# ══════════════════════════════════════════════════════════════════════════════
# SETUP PANEL
# ══════════════════════════════════════════════════════════════════════════════
with st.expander(
    "⚙️ Alert Settings — API Key, Gmail & Contacts",
    expanded=not bool(cfg.get("gmail") or cfg.get("api_key")),
):
    st.caption(
        "সেটিংস `alert_settings.json`-এ সেভ হবে (git-ignored — শুধু আপনার ডিভাইসে থাকবে)।"
    )
    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("##### 📱 BulkSMSBD (SMS)")
        s_api   = st.text_input("API Key",   value=cfg.get("api_key", ""),
                                type="password", key="s_api")
        s_sid   = st.text_input("Sender ID", value=cfg.get("sender_id", ""),
                                placeholder="HaorAlert", key="s_sid")
        st.caption(
            "API key: [bulksmsbd.net](http://bulksmsbd.net) → My Account → API Keys"
        )

    with sc2:
        st.markdown("##### 📧 Gmail (Email)")
        s_gmail = st.text_input("Gmail Address",      value=cfg.get("gmail", ""),
                                placeholder="you@gmail.com", key="s_gmail")
        s_pass  = st.text_input("Gmail App Password", value=cfg.get("password", ""),
                                type="password",
                                placeholder="xxxx xxxx xxxx xxxx", key="s_pass",
                                help="myaccount.google.com → Security → App Passwords → Generate")

    st.markdown("##### 👥 Contacts CSV")
    dl_col, up_col = st.columns(2)
    with dl_col:
        st.download_button(
            "📥 Download sample contacts.csv",
            data=_SAMPLE_CSV.encode("utf-8-sig"),
            file_name="contacts_sample.csv",
            mime="text/csv",
        )
    with up_col:
        uploaded = st.file_uploader(
            "Upload your contacts.csv",
            type=["csv"],
            key="contacts_up",
            help="Required columns: name, phone, email (any can be blank)",
        )

    contacts_preview = list(cfg.get("contacts", []))
    if uploaded is not None:
        try:
            reader = csv.DictReader(
                io.StringIO(uploaded.read().decode("utf-8", errors="replace"))
            )
            contacts_preview = [dict(r) for r in reader]
            st.success(f"✅ {len(contacts_preview)} contacts loaded.")
        except Exception as exc:
            st.error(f"CSV parse error: {exc}")

    if contacts_preview:
        st.dataframe(
            pd.DataFrame(contacts_preview), use_container_width=True, height=140
        )

    col_save, col_rm = st.columns(2)
    with col_save:
        if st.button("💾 Save Settings", type="primary",
                     use_container_width=True, key="btn_save"):
            new_cfg = {
                "api_key":   s_api.strip(),
                "sender_id": s_sid.strip(),
                "gmail":     s_gmail.strip(),
                "password":  s_pass.strip(),
                "contacts":  contacts_preview,
            }
            save_settings(new_cfg)
            st.success("✅ Saved to `alert_settings.json`.")
            st.rerun()

    with col_rm:
        if st.button("🗑️ Remove Settings", use_container_width=True, key="btn_rm"):
            if _SETTINGS_PATH.exists():
                _SETTINGS_PATH.unlink()
            st.warning("Settings file removed.")
            st.rerun()

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_manual, tab_auto = st.tabs(["📤 Manual Alerts", "🤖 Automated"])


# ─────────────────────────── TAB 1 : MANUAL ──────────────────────────────────
with tab_manual:
    st.markdown("### 📤 Manual Alert Dispatch")

    # Inputs
    ic1, ic2 = st.columns(2)
    with ic1:
        m_prob  = st.slider("Flood Probability (%)", 0, 100, 71, key="m_prob")
        m_barak = st.number_input(
            "Barak Discharge (m³/s)", 0.0, 20000.0, 3200.0,
            step=100.0, key="m_barak",
        )
    with ic2:
        m_phone = st.text_input(
            "Test phone (single number)", placeholder="01712345678", key="m_phone"
        )
        m_email_to = st.text_input(
            "Test email (single address)", placeholder="farmer@gmail.com", key="m_email_to"
        )

    # Derived
    m_risk         = prob_to_risk(m_prob)
    m_label, m_col = risk_badge(m_risk)
    m_sms          = build_sms(m_risk, m_barak, m_prob)
    m_subj, m_body = build_email_body(m_risk, m_prob, m_barak, m_sms)
    month_now      = datetime.now().month
    boro_note      = "🌾 বোরো মৌসুম" if month_now in [1, 2, 3, 4, 5] else "🌿 বোরো মৌসুম নয়"

    # Risk badge
    st.markdown(
        f"<div style='padding:10px 18px;border-radius:8px;"
        f"border-left:6px solid {m_col};background:rgba(0,0,0,0.06);margin:8px 0'>"
        f"<b style='color:{m_col};font-size:1.1rem'>{m_label}</b>"
        f"&nbsp;·&nbsp;{m_prob}%"
        f"&nbsp;·&nbsp;Barak: {m_barak:,.0f} m³/s"
        f"&nbsp;·&nbsp;<span style='font-size:0.85rem;opacity:0.75'>{boro_note}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # SMS preview + WhatsApp copy
    prev_col, wp_col = st.columns([3, 2])
    with prev_col:
        st.markdown("**📱 SMS Preview (Bengali):**")
        st.code(m_sms, language=None)
    with wp_col:
        st.markdown("**💬 WhatsApp Copy:**")
        st.text_area(
            "whatsapp",
            value=m_sms,
            height=170,
            key="wp_copy",
            label_visibility="collapsed",
        )
        st.caption("Select all → Copy → Paste into WhatsApp group")

    st.divider()

    # Action buttons
    contacts = cfg.get("contacts", [])
    n_sms_c  = sum(1 for c in contacts if str(c.get("phone", "")).strip())
    n_eml_c  = sum(1 for c in contacts if str(c.get("email", "")).strip())

    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        sms_ready = bool(cfg.get("api_key") and cfg.get("sender_id") and m_phone.strip())
        if st.button("📱 Test SMS", use_container_width=True,
                     disabled=not sms_ready, key="btn_test_sms"):
            with st.spinner("Sending SMS..."):
                ok, det = send_sms(
                    m_phone.strip(), m_sms, cfg["api_key"], cfg["sender_id"]
                )
            (st.success if ok else st.error)(
                f"{'✅ Sent' if ok else '❌ Failed'} → {m_phone} | {det[:80]}"
            )
            append_history("SMS", m_phone, m_risk, ok, det)

    with btn2:
        eml_ready = bool(cfg.get("gmail") and cfg.get("password") and m_email_to.strip())
        if st.button("📧 Test Email", use_container_width=True,
                     disabled=not eml_ready, key="btn_test_email"):
            with st.spinner("Sending email via Gmail SMTP..."):
                ok, det = send_email(
                    m_email_to.strip(), m_subj, m_body,
                    cfg["gmail"], cfg["password"],
                )
            (st.success if ok else st.error)(
                f"{'✅ Sent' if ok else '❌ Failed'} → {m_email_to} | {det[:80]}"
            )
            append_history("Email", m_email_to, m_risk, ok, det)

    with btn3:
        all_ready = bool(contacts and (cfg.get("api_key") or cfg.get("gmail")))
        all_label = f"🚨 Send All  ({n_sms_c} SMS · {n_eml_c} email)"
        if st.button(all_label, use_container_width=True, type="primary",
                     disabled=not all_ready, key="btn_send_all"):
            sent_s = sent_e = failed = 0
            prog = st.progress(0.0, text="Sending...")
            for i, contact in enumerate(contacts):
                phone = str(contact.get("phone", "")).strip()
                email = str(contact.get("email", "")).strip()
                name  = str(contact.get("name",  "farmer"))
                if phone and cfg.get("api_key"):
                    ok, det = send_sms(phone, m_sms, cfg["api_key"], cfg["sender_id"])
                    append_history("SMS", f"{name} ({phone})", m_risk, ok, det)
                    if ok: sent_s += 1
                    else:  failed += 1
                    time.sleep(0.3)
                if email and cfg.get("gmail"):
                    ok, det = send_email(
                        email, m_subj, m_body, cfg["gmail"], cfg["password"]
                    )
                    append_history("Email", f"{name} ({email})", m_risk, ok, det)
                    if ok: sent_e += 1
                    else:  failed += 1
                prog.progress((i + 1) / len(contacts),
                              text=f"{i+1}/{len(contacts)} contacts...")
            prog.empty()
            st.success(
                f"✅ Done — SMS: **{sent_s}** · Email: **{sent_e}** · Failed: **{failed}**"
            )

    # Status hints
    missing = []
    if not cfg.get("api_key"):  missing.append("BulkSMSBD API key (SMS disabled)")
    if not cfg.get("gmail"):    missing.append("Gmail credentials (email disabled)")
    if not contacts:            missing.append("contacts CSV (Send All disabled)")
    if missing:
        st.caption("⚠️  Missing: " + " · ".join(missing) + " — add in ⚙️ Settings above.")

    # History
    st.divider()
    st.markdown("### 📋 Alert History (this session)")
    history = st.session_state["alert_history"]
    if history:
        st.dataframe(
            pd.DataFrame(history[::-1]),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("🗑️ Clear history", key="btn_clear_hist"):
            st.session_state["alert_history"] = []
            st.rerun()
    else:
        st.caption("No alerts sent yet this session.")


# ─────────────────────────── TAB 2 : AUTOMATED ───────────────────────────────
with tab_auto:
    st.markdown("### 🤖 Automated Alert")
    st.info(
        "When **ON**, alerts are dispatched automatically whenever the flood "
        "probability from the Prediction page exceeds the threshold. "
        "A 1-hour cooldown prevents duplicate sends."
    )

    ac1, ac2 = st.columns([1, 2])
    with ac1:
        auto_on = st.toggle(
            "AUTO-SEND",
            value=st.session_state["auto_on"],
            key="auto_toggle",
        )
        st.session_state["auto_on"] = auto_on
        threshold = st.slider(
            "Trigger threshold (%)", 50, 95, 75, key="auto_thresh"
        )

    with ac2:
        if auto_on:
            st.success(
                f"✅ AUTO-SEND **ON** — triggers when probability > **{threshold}%**"
            )
        else:
            st.warning("⏸️ AUTO-SEND **OFF**")
        last_sent = st.session_state["auto_last_sent"]
        st.caption(
            f"Last auto-alert: **{last_sent}**" if last_sent
            else "No auto-alert sent this session."
        )

    st.divider()
    st.markdown("##### 🧪 Simulate / Fire Manually")

    ta1, ta2 = st.columns(2)
    with ta1:
        auto_prob  = st.slider(
            "Current probability (%)", 0, 100, 78, key="auto_prob"
        )
        auto_barak = st.number_input(
            "Barak discharge (m³/s)", 0.0, 20000.0, 7200.0,
            step=100.0, key="auto_barak",
        )
    with ta2:
        auto_risk       = prob_to_risk(auto_prob)
        auto_sms        = build_sms(auto_risk, auto_barak, auto_prob)
        auto_label, _ac = risk_badge(auto_risk)
        st.markdown(f"**Would send to {len(cfg.get('contacts', []))} contacts:**")
        st.code(auto_sms, language=None)

    will_trigger = auto_on and (auto_prob >= threshold)
    contacts_auto = cfg.get("contacts", [])

    if will_trigger:
        st.error(
            f"🚨 Condition MET — {auto_prob}% ≥ {threshold}% threshold"
        )
    elif auto_on:
        st.success(
            f"✅ Monitoring — {auto_prob}% < {threshold}%, no trigger"
        )
    else:
        st.info("Toggle AUTO-SEND ON to activate monitoring.")

    has_contacts = bool(contacts_auto)
    has_channels = bool(cfg.get("api_key") or cfg.get("gmail"))
    fire_disabled = not (will_trigger and has_contacts and has_channels)

    if st.button(
        f"🚨 Fire Now  ({len(contacts_auto)} contacts)",
        type="primary",
        disabled=fire_disabled,
        key="btn_auto_fire",
    ):
        auto_subj, auto_body = build_email_body(
            auto_risk, auto_prob, auto_barak, auto_sms
        )
        sent_s = sent_e = failed = 0
        prog = st.progress(0.0)
        for i, contact in enumerate(contacts_auto):
            phone = str(contact.get("phone", "")).strip()
            email = str(contact.get("email", "")).strip()
            name  = str(contact.get("name",  "farmer"))
            if phone and cfg.get("api_key"):
                ok, det = send_sms(
                    phone, auto_sms, cfg["api_key"], cfg["sender_id"]
                )
                append_history("AUTO-SMS", f"{name} ({phone})", auto_risk, ok, det)
                if ok: sent_s += 1
                else:  failed += 1
                time.sleep(0.3)
            if email and cfg.get("gmail"):
                ok, det = send_email(
                    email, auto_subj, auto_body, cfg["gmail"], cfg["password"]
                )
                append_history(
                    "AUTO-Email", f"{name} ({email})", auto_risk, ok, det
                )
                if ok: sent_e += 1
                else:  failed += 1
            prog.progress((i + 1) / len(contacts_auto))
        prog.empty()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["auto_last_sent"] = ts
        st.success(
            f"✅ Auto-alert fired at {ts} — "
            f"SMS: **{sent_s}** · Email: **{sent_e}** · Failed: **{failed}**"
        )

    if not has_contacts:
        st.caption("⚠️ Upload contacts CSV in ⚙️ Settings to enable Fire Now.")
    if not has_channels:
        st.caption("⚠️ Add API key or Gmail credentials in ⚙️ Settings.")

    st.divider()
    st.markdown("##### 📋 Automated Alert History (this session)")
    auto_hist = [
        h for h in st.session_state["alert_history"]
        if h["Channel"].startswith("AUTO")
    ]
    if auto_hist:
        st.dataframe(
            pd.DataFrame(auto_hist[::-1]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No automated alerts fired this session.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HaorFloodAlert — SMS via BulkSMSBD (bulksmsbd.net) · "
    "Email via Gmail SMTP SSL (port 465) | Sunamganj Haor, Bangladesh"
)

with st.sidebar:
    contacts_sb = cfg.get("contacts", [])
    n_ph = sum(1 for c in contacts_sb if str(c.get("phone", "")).strip())
    n_em = sum(1 for c in contacts_sb if str(c.get("email", "")).strip())
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;color:#888;text-align:center;line-height:1.8'>"
        f"🚨 <b>Alert Status</b><br>"
        f"Contacts: {len(contacts_sb)}"
        f" ({n_ph} SMS · {n_em} email)<br>"
        f"Auto-send: {'✅ ON' if st.session_state.get('auto_on') else '⏸️ OFF'}<br>"
        f"Session alerts: {len(st.session_state['alert_history'])}<br>"
        f"{'─'*22}<br>"
        f"🌊 <b>HaorFloodAlert v2.0</b><br>"
        f"© 2026 Salma Hoque Talukdar Koli<br>"
        f"RTM Al-Kabir Technical University<br>"
        f"CSE Thesis Project"
        f"</div>",
        unsafe_allow_html=True,
    )
