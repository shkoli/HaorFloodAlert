"""
email_alert.py — HaorFloodAlert Gmail SMTP alert sender.

Usage:
    from streamlit_app.alerts.email_alert import send_flood_alert
    ok, msg = send_flood_alert(
        flood_prob=0.87,
        upstream_vv=-9.2,
        rainfall=145.0,
        sender="you@gmail.com",
        password="xxxx xxxx xxxx xxxx",
        recipient="target@gmail.com",
    )
"""

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# ── Risk classification ───────────────────────────────────────────────────────

def _classify(flood_prob: float) -> dict:
    if flood_prob >= 0.85:
        return {
            "level": "EXTREME",
            "label": "অত্যন্ত বিপদজনক (EXTREME)",
            "color": "#CC0000",
            "bg":    "#FFF0F0",
            "action_bn": "এখনই নিরাপদ স্থানে যান। শিশু, বয়স্ক ও গবাদিপশু সরান।",
            "action_en": "EVACUATE IMMEDIATELY. Move children, elderly, and livestock to safety.",
            "emoji": "🔴",
        }
    if flood_prob >= 0.65:
        return {
            "level": "HIGH",
            "label": "উচ্চ ঝুঁকি (HIGH)",
            "color": "#E65C00",
            "bg":    "#FFF5EC",
            "action_bn": "জরুরি জিনিসপত্র গুছিয়ে নিন। উঁচু জায়গায় যাওয়ার পরিকল্পনা করুন।",
            "action_en": "Prepare emergency supplies. Plan evacuation route.",
            "emoji": "🟠",
        }
    if flood_prob >= 0.40:
        return {
            "level": "MEDIUM",
            "label": "মাঝারি ঝুঁকি (MEDIUM)",
            "color": "#B8860B",
            "bg":    "#FFFBEC",
            "action_bn": "সতর্ক থাকুন। জিনিসপত্র উঁচুতে রাখুন।",
            "action_en": "Stay alert. Move valuables to higher ground.",
            "emoji": "🟡",
        }
    return {
        "level": "LOW",
        "label": "স্বাভাবিক (LOW)",
        "color": "#1A7A4A",
        "bg":    "#F0FFF8",
        "action_bn": "এখন কোনো বড় বন্যার ঝুঁকি নেই। নিয়মিত পর্যবেক্ষণ করুন।",
        "action_en": "No major flood threat. Continue routine monitoring.",
        "emoji": "🟢",
    }


def _upstream_note(upstream_vv: float) -> str:
    """Human-readable interpretation of upstream Barak river VV backscatter."""
    if upstream_vv <= -12.0:
        return "Very high upstream discharge detected — flood likely within ~36 hours."
    if upstream_vv <= -10.0:
        return "Elevated upstream water level — monitor closely."
    if upstream_vv <= -8.0:
        return "Moderate upstream signal — minor risk of downstream inundation."
    return "Normal upstream conditions."


# ── HTML email body ───────────────────────────────────────────────────────────

def _build_html(flood_prob: float, upstream_vv: float, rainfall: float,
                risk: dict, timestamp: str) -> str:
    prob_pct = f"{flood_prob * 100:.1f}%"
    upstream_note = _upstream_note(upstream_vv)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HaorFloodAlert</title>
</head>
<body style="margin:0;padding:0;background:#F4F4F4;font-family:Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" bgcolor="#F4F4F4">
<tr><td align="center" style="padding:30px 10px;">

  <!-- Card -->
  <table width="560" cellpadding="0" cellspacing="0"
         style="background:#FFFFFF;border-radius:10px;
                border:2px solid {risk['color']};overflow:hidden;">

    <!-- Header -->
    <tr>
      <td style="background:{risk['color']};padding:20px 28px;">
        <h1 style="margin:0;color:#FFFFFF;font-size:22px;letter-spacing:0.5px;">
          {risk['emoji']} HaorFloodAlert — বন্যা সতর্কতা
        </h1>
        <p style="margin:6px 0 0;color:#FFFFFF;opacity:0.92;font-size:14px;">
          Sunamganj Haor Early Warning System · {timestamp}
        </p>
      </td>
    </tr>

    <!-- Risk banner -->
    <tr>
      <td style="background:{risk['bg']};padding:18px 28px;
                 border-bottom:1px solid {risk['color']}30;">
        <p style="margin:0;font-size:28px;font-weight:bold;color:{risk['color']};">
          {prob_pct} Flood Probability
        </p>
        <p style="margin:4px 0 0;font-size:16px;color:{risk['color']};font-weight:600;">
          {risk['label']}
        </p>
      </td>
    </tr>

    <!-- Sensor data -->
    <tr>
      <td style="padding:20px 28px;">
        <h2 style="margin:0 0 12px;font-size:15px;color:#333;
                   text-transform:uppercase;letter-spacing:0.5px;">
          Sensor Readings
        </h2>
        <table width="100%" cellpadding="8" cellspacing="0"
               style="border-collapse:collapse;font-size:14px;">
          <tr style="background:#F8F9FA;">
            <td style="border:1px solid #DDD;padding:10px 14px;color:#555;">
              Rainfall (7-day cumulative)
            </td>
            <td style="border:1px solid #DDD;padding:10px 14px;
                       font-weight:bold;color:#222;">
              {rainfall:.1f} mm
            </td>
          </tr>
          <tr>
            <td style="border:1px solid #DDD;padding:10px 14px;color:#555;">
              Upstream Barak VV Backscatter
            </td>
            <td style="border:1px solid #DDD;padding:10px 14px;
                       font-weight:bold;color:#222;">
              {upstream_vv:.2f} dB &nbsp;
              <span style="font-weight:normal;color:#666;font-size:12px;">
                ({upstream_note})
              </span>
            </td>
          </tr>
          <tr style="background:#F8F9FA;">
            <td style="border:1px solid #DDD;padding:10px 14px;color:#555;">
              Ensemble Flood Probability
            </td>
            <td style="border:1px solid #DDD;padding:10px 14px;
                       font-weight:bold;color:{risk['color']};">
              {prob_pct}
            </td>
          </tr>
          <tr>
            <td style="border:1px solid #DDD;padding:10px 14px;color:#555;">
              Upstream Lead Time Estimate
            </td>
            <td style="border:1px solid #DDD;padding:10px 14px;
                       font-weight:bold;color:#222;">
              ~36 hours (Barak → Surma → Haor)
            </td>
          </tr>
        </table>
      </td>
    </tr>

    <!-- Recommended action -->
    <tr>
      <td style="padding:0 28px 20px;">
        <div style="background:{risk['bg']};border-left:4px solid {risk['color']};
                    border-radius:4px;padding:14px 16px;">
          <p style="margin:0 0 6px;font-size:15px;font-weight:bold;
                    color:{risk['color']};">
            Recommended Action
          </p>
          <p style="margin:0 0 8px;font-size:14px;color:#222;">
            {risk['action_en']}
          </p>
          <p style="margin:0;font-size:14px;color:#444;
                    font-family:'Noto Sans Bengali',Arial,sans-serif;">
            {risk['action_bn']}
          </p>
        </div>
      </td>
    </tr>

    <!-- Divider -->
    <tr><td style="padding:0 28px;"><hr style="border:none;border-top:1px solid #EEE;"></td></tr>

    <!-- Footer -->
    <tr>
      <td style="padding:16px 28px;">
        <p style="margin:0;font-size:12px;color:#888;line-height:1.6;">
          <strong>HaorFloodAlert</strong> · RTM Al-Kabir Technical University<br>
          Ensemble: RF×0.45 + XGBoost×0.35 + LSTM×0.20 ·
          LOOCV accuracy 88.9% on 72 real Sentinel-1 events<br>
          Always verify with
          <a href="http://www.ffwc.gov.bd" style="color:#888;">FFWC</a>
          before taking evacuation decisions.<br>
          <em>This is an automated research prototype — not an official government alert.</em>
        </p>
      </td>
    </tr>

  </table>
  <!-- /Card -->

</td></tr>
</table>
</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────

def send_flood_alert(
    flood_prob: float,
    upstream_vv: float,
    rainfall: float,
    sender: str,
    password: str,
    recipient: str,
) -> tuple[bool, str]:
    """
    Send a flood alert email via Gmail SMTP (SSL port 465).

    Args:
        flood_prob:   Ensemble flood probability, 0.0–1.0.
        upstream_vv:  Upstream Barak river VV backscatter in dB (e.g. -9.2).
        rainfall:     7-day cumulative rainfall in mm.
        sender:       Gmail address to send from.
        password:     Gmail App Password (16 chars, spaces allowed).
        recipient:    Destination email address.

    Returns:
        (True, "sent")  on success.
        (False, reason) on failure.
    """
    risk = _classify(flood_prob)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = (
        f"{risk['emoji']} {risk['level']} Flood Alert — "
        f"Sunamganj Haor ({flood_prob*100:.0f}%) · {timestamp}"
    )
    html = _build_html(flood_prob, upstream_vv, rainfall, risk, timestamp)

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender, password.replace(" ", ""))
            server.sendmail(sender, recipient, msg.as_string())

        return True, "sent"

    except smtplib.SMTPAuthenticationError:
        return False, (
            "Gmail authentication failed. "
            "Check that you are using an App Password (not your account password). "
            "Generate one at: myaccount.google.com/apppasswords"
        )
    except smtplib.SMTPRecipientsRefused:
        return False, f"Recipient address rejected by Gmail: {recipient}"
    except smtplib.SMTPException as exc:
        return False, f"SMTP error: {exc}"
    except OSError as exc:
        return False, f"Network error: {exc}"
