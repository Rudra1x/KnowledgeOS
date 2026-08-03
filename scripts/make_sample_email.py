# scripts/make_sample_email.py

from email.message import EmailMessage
from pathlib import Path

msg = EmailMessage()
msg["Subject"] = "Q4 planning review - action items"
msg["From"]    = "Alice Chen <alice.chen@example.com>"
msg["To"]      = "Bob Rao <bob.rao@example.com>, Carol Singh <carol.singh@example.com>"
msg["Cc"]      = "team-leads@example.com"
msg["Date"]    = "Wed, 15 Nov 2023 10:30:00 -0500"

plain = """Hi Bob and Carol,

Thanks for joining the Q4 planning review yesterday. Summarizing the action items:

1. Bob to finalize the retrieval benchmark results by Friday
2. Carol to draft the customer-facing changelog by end of next week
3. All: review the attached architecture doc before the deep dive on Monday

Let me know if I missed anything.

Best,
Alice"""

html = """<html><body>
<p>Hi Bob and Carol,</p>
<p>Thanks for joining the Q4 planning review yesterday. Summarizing the action items:</p>
<ol>
  <li>Bob to finalize the retrieval benchmark results by Friday</li>
  <li>Carol to draft the customer-facing changelog by end of next week</li>
  <li>All: review the attached architecture doc before the deep dive on Monday</li>
</ol>
<p>Let me know if I missed anything.</p>
<p>Best,<br>Alice</p>
</body></html>"""

msg.set_content(plain)
msg.add_alternative(html, subtype="html")

# Fake attachment
msg.add_attachment(
    b"Architecture doc placeholder content.",
    maintype    = "application",
    subtype     = "pdf",
    filename    = "architecture.pdf",
)

out = Path("D:/Rudraksh/KnowledgeOS/scripts/sample.eml")
out.write_bytes(bytes(msg))
print(f"Wrote {out} ({out.stat().st_size} bytes)")