# eval/report_generator.py

import json
from datetime import datetime


def generate_report(results: dict, output_path: str = "eval_report.html") -> str:
    """
    Generate a standalone HTML evaluation report from PipelineEvaluator results.

    Parameters
    ----------
    results     : dict   output from PipelineEvaluator.evaluate()
    output_path : str    where to save the HTML file

    Returns the output path.
    """
    html = _build_html(results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def _build_html(results: dict) -> str:
    agg   = results["aggregate"]
    by_t  = results["by_type"]
    by_d  = results["by_difficulty"]
    pq    = results["per_query"]
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

    def pct(v):
        if v is None:
            return "N/A"
        return f"{v*100:.1f}%"

    def color(v, lo=0.6, hi=0.85):
        if v is None:
            return "#6b7280"
        if v >= hi:
            return "#16a34a"
        if v >= lo:
            return "#ca8a04"
        return "#dc2626"

    # --- Metric cards ---
    metrics = [
        ("recall@1",    agg["recall@1"],    "Retrieval"),
        ("nDCG@3",      agg["ndcg@3"],      "Retrieval"),
        ("MRR",         agg["mrr"],         "Retrieval"),
        ("Faithfulness",agg["faithfulness"],"Generation"),
        ("Relevance",   agg["relevance"],   "Generation"),
        ("Cit. Coverage",agg["citation_coverage"],"Generation"),
        ("Neg. Decline",agg["decline_rate"],"Robustness"),
    ]

    cards_html = ""
    for label, val, category in metrics:
        c    = color(val)
        disp = pct(val) if val is not None else "N/A"
        cards_html += f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{c}">{disp}</div>
          <div class="card-cat">{category}</div>
        </div>"""

    # --- By type table ---
    type_rows = ""
    for qt, m in sorted(by_t.items()):
        type_rows += f"""
        <tr>
          <td><span class="badge badge-{qt}">{qt}</span></td>
          <td style="color:{color(m['recall@1'])}">{pct(m['recall@1'])}</td>
          <td style="color:{color(m['ndcg@3'])}">{pct(m['ndcg@3'])}</td>
          <td style="color:{color(m['faithfulness'])}">{pct(m['faithfulness'])}</td>
          <td style="color:{color(m['relevance'])}">{pct(m['relevance'])}</td>
          <td>{m['n']}</td>
        </tr>"""

    # --- By difficulty table ---
    diff_rows = ""
    for d, m in sorted(by_d.items()):
        diff_rows += f"""
        <tr>
          <td><b>{d}</b></td>
          <td style="color:{color(m['recall@1'])}">{pct(m['recall@1'])}</td>
          <td style="color:{color(m['ndcg@3'])}">{pct(m['ndcg@3'])}</td>
          <td style="color:{color(m['faith'])}">{pct(m['faith'])}</td>
          <td style="color:{color(m['rel'])}">{pct(m['rel'])}</td>
          <td>{m['n']}</td>
        </tr>"""

    # --- Per-query table ---
    pq_rows = ""
    for q in pq:
        if q["is_negative"]:
            status = "DECLINED" if q.get("declined") else "WRONG"
            sc     = "#16a34a" if q.get("declined") else "#dc2626"
            pq_rows += f"""
        <tr>
          <td class="query-text">{q['query'][:55]}</td>
          <td><span class="badge badge-{q['query_type']}">{q['query_type']}</span></td>
          <td><span class="badge badge-{q['difficulty']}">{q['difficulty']}</span></td>
          <td colspan="4" style="text-align:center;color:{sc}">
            NEGATIVE — {status}</td>
        </tr>"""
        else:
            r1c   = color(q.get("recall@1"))
            fc    = color(q.get("faithfulness"))
            rc    = color(q.get("relevance"))
            ans   = (q.get("answer","")[:120] + "...").replace("<","&lt;").replace(">","&gt;")
            pq_rows += f"""
        <tr>
          <td class="query-text" title="{q['query']}">{q['query'][:55]}</td>
          <td><span class="badge badge-{q['query_type']}">{q['query_type']}</span></td>
          <td><span class="badge badge-{q['difficulty']}">{q['difficulty']}</span></td>
          <td style="color:{r1c};font-weight:bold">{pct(q.get('recall@1'))}</td>
          <td style="color:{fc}">{pct(q.get('faithfulness'))}</td>
          <td style="color:{rc}">{pct(q.get('relevance'))}</td>
          <td class="answer-snippet" title="{ans}">{ans[:80]}...</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KnowledgeOS Eval — {results['run_name']}</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text);
    padding: 2rem; line-height: 1.5;
  }}
  h1 {{ font-size: 1.8rem; color: var(--accent); margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 2rem 0 0.75rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 0.75rem; padding: 1.25rem 1.5rem;
    min-width: 130px; flex: 1;
  }}
  .card-label {{ font-size: 0.75rem; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.06em; }}
  .card-value {{ font-size: 2rem; font-weight: 700; margin: 0.25rem 0; }}
  .card-cat   {{ font-size: 0.7rem; color: var(--muted); }}
  table {{
    width: 100%; border-collapse: collapse;
    background: var(--surface); border-radius: 0.5rem;
    overflow: hidden; font-size: 0.875rem;
  }}
  th {{
    background: #0f172a; color: var(--muted);
    padding: 0.6rem 0.75rem; text-align: left;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  td {{ padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #1a2a3a; }}
  .badge {{
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 9999px; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .badge-factoid    {{ background:#1e3a5f; color:#93c5fd; }}
  .badge-comparison {{ background:#1e3a2a; color:#86efac; }}
  .badge-thematic   {{ background:#3a2a1e; color:#fcd34d; }}
  .badge-negative   {{ background:#3a1e1e; color:#fca5a5; }}
  .badge-easy       {{ background:#1a2a1a; color:#4ade80; }}
  .badge-medium     {{ background:#2a2a1a; color:#facc15; }}
  .badge-hard       {{ background:#2a1a1a; color:#f87171; }}
  .query-text       {{ max-width: 280px; white-space: nowrap;
                       overflow: hidden; text-overflow: ellipsis; }}
  .answer-snippet   {{ max-width: 300px; white-space: nowrap;
                       overflow: hidden; text-overflow: ellipsis;
                       color: var(--muted); font-size: 0.8rem; }}
  .section {{ margin-top: 2rem; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  footer {{ margin-top: 3rem; color: var(--muted);
            font-size: 0.75rem; text-align: center; }}
</style>
</head>
<body>

<h1>KnowledgeOS Evaluation Report</h1>
<div class="meta">
  Run: <b>{results['run_name']}</b> &nbsp;|&nbsp;
  Queries: <b>{results['n_queries']}</b> &nbsp;|&nbsp;
  Total time: <b>{results['total_s']}s</b> &nbsp;|&nbsp;
  Generated: <b>{ts}</b>
</div>

<h2>Summary</h2>
<div class="cards">{cards_html}</div>

<div class="two-col section">
  <div>
    <h2>By Query Type</h2>
    <table>
      <thead><tr>
        <th>Type</th><th>recall@1</th><th>nDCG@3</th>
        <th>Faith.</th><th>Rel.</th><th>N</th>
      </tr></thead>
      <tbody>{type_rows}</tbody>
    </table>
  </div>
  <div>
    <h2>By Difficulty</h2>
    <table>
      <thead><tr>
        <th>Difficulty</th><th>recall@1</th><th>nDCG@3</th>
        <th>Faith.</th><th>Rel.</th><th>N</th>
      </tr></thead>
      <tbody>{diff_rows}</tbody>
    </table>
  </div>
</div>

<div class="section">
  <h2>Per-Query Detail</h2>
  <table>
    <thead><tr>
      <th>Query</th><th>Type</th><th>Difficulty</th>
      <th>recall@1</th><th>Faith.</th><th>Rel.</th><th>Answer</th>
    </tr></thead>
    <tbody>{pq_rows}</tbody>
  </table>
</div>

<footer>KnowledgeOS &mdash; Built milestone by milestone &mdash; {ts}</footer>
</body>
</html>"""