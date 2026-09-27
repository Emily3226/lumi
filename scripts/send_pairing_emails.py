"""
One-off script to send the Fall 2026 pairing-announcement emails as
auxilium.mentorship@gmail.com via Gmail SMTP.

Resend (used by the admin panel's normal email sender) can't be used for this
From address: it can only send from a domain you've verified there via DNS,
and gmail.com can't be verified that way. See
api.pairing_emails.send_rendered_email_via_gmail for details.

Usage:
    python scripts/send_pairing_emails.py                          # dry run: render + print counts only, no credentials needed
    python scripts/send_pairing_emails.py --test-address me@x.com  # real send, but every email redirected to this one address
    python scripts/send_pairing_emails.py --confirm SEND           # actually sends every email to its real recipient(s)

Requires in .env (or, in CI, as environment variables) - only read when actually sending:
    AUXILIUM_GMAIL_ADDRESS=auxilium.mentorship@gmail.com
    AUXILIUM_GMAIL_APP_PASSWORD=<16-character app password from myaccount.google.com/apppasswords>

Optional in .env:
    PAIRINGS_CSV_PATH=<path to the pairings CSV> (defaults to the Fall 2026 file in Downloads)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from api.pairing_emails import parse_pairings_csv, render_emails, send_rendered_email_via_gmail

DEFAULT_CSV_PATH = r"C:\Users\ezhan\Downloads\NEW MENTOR_MENTEE PAIRINGS - Fall 2026 - pairings.csv"
SUBJECT_TEMPLATE = "Mentor-Mentee Pairings"


def load_templates() -> tuple[str, str]:
    admin_html_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "admin.html"
    )
    html = open(admin_html_path, encoding="utf-8").read()
    mentor_tmpl = re.search(r"const PE_DEFAULT_MENTOR_TEMPLATE = `([\s\S]*?)`;", html).group(1)
    mentee_tmpl = re.search(r"const PE_DEFAULT_MENTEE_TEMPLATE = `([\s\S]*?)`;", html).group(1)
    return mentor_tmpl, mentee_tmpl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", default="", help="Must be exactly SEND to actually send to real recipients.")
    parser.add_argument(
        "--test-address",
        default="",
        help="If set, every email is redirected to this single address instead of its real recipient(s) - "
        "for verifying the send path (subject/body/from) without touching real mentors/mentees.",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between sends (avoids Gmail rate limits).")
    parser.add_argument("--csv", default=os.environ.get("PAIRINGS_CSV_PATH", DEFAULT_CSV_PATH))
    args = parser.parse_args()

    mentor_tmpl, mentee_tmpl = load_templates()
    csv_text = open(args.csv, encoding="utf-8-sig").read()
    mentors = parse_pairings_csv(csv_text)
    rendered = render_emails(mentors, mentor_tmpl, mentee_tmpl, SUBJECT_TEMPLATE, {})

    print(f"{len(mentors)} mentor groups -> {len(rendered)} emails")

    if not args.test_address and args.confirm != "SEND":
        print("Dry run only (pass --test-address you@example.com to test, or --confirm SEND to send for real). Nothing was sent.")
        return

    gmail_address = os.environ["AUXILIUM_GMAIL_ADDRESS"]
    gmail_app_password = os.environ["AUXILIUM_GMAIL_APP_PASSWORD"]
    print(f"Sending from {gmail_address}" + (f" (TEST MODE: all redirected to {args.test_address})" if args.test_address else ""))

    ok_count = 0
    for email in rendered:
        if args.test_address:
            email.to = [args.test_address]
            email.subject = f"[TEST - {email.recipient_type}] {email.subject}"
        ok, detail = send_rendered_email_via_gmail(email, gmail_address, gmail_app_password)
        print(f"{'OK  ' if ok else 'FAIL'} {email.recipient_type:6} {email.mentor_name:25} -> {email.to} [{detail}]")
        if ok:
            ok_count += 1
        time.sleep(args.delay)

    print(f"Done: {ok_count}/{len(rendered)} sent successfully.")


if __name__ == "__main__":
    main()
