import os
import smtplib
from email.message import EmailMessage

SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER).strip()
SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1") == "1"
SMTP_SSL = os.environ.get("SMTP_SSL", "0") == "1"


def email_configured() -> bool:
    return bool(SMTP_HOST and SMTP_PORT and SMTP_FROM)


def send_email(to_email: str, subject: str, text_body: str) -> bool:
    if not email_configured():
        missing = []
        if not SMTP_HOST:
            missing.append("SMTP_HOST")
        if not SMTP_FROM:
            missing.append("SMTP_FROM")
        print("Email disabled: missing SMTP config:", ", ".join(missing) if missing else "unknown")
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_body)

    try:
        if SMTP_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
                if SMTP_USER:
                    smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
                if SMTP_STARTTLS:
                    smtp.starttls()
                if SMTP_USER:
                    smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
        return True
    except Exception as e:
        print("Email send failed:", e)
        return False
