#!/usr/bin/env python3
"""
Project notifier for PancsVriend / Schelling LLM runs.

Usage:
    ./notify.py "Subject" "Body"
"""
import os
import smtplib
import sys
from email.message import EmailMessage

GMAIL_USER   = "dr.duus@gmail.com"
GMAIL_APP_PW = os.environ["GMAIL_APP_PW"]

RECIPIENTS = [
    "dr.duus@gmail.com",
    "siyer5@binghamton.edu",
]


def send(subject, body):
    msg = EmailMessage()
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(RECIPIENTS)
    msg["Subject"] = subject
    msg.set_content(
        body + "\n\n---\nThis is an automatically generated message from the PancsVriend simulation server (Linux machine). No reply needed."
    )
    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.starttls()
        s.login(GMAIL_USER, GMAIL_APP_PW)
        s.send_message(msg)


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "PancsVriend notification"
    body    = sys.argv[2] if len(sys.argv) > 2 else ""
    send(subject, body)
    print("Sent.")
