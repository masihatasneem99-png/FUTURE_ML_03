"""
report.py:
Visual HTML report generator for the Resume Screening System.

Takes the ranked DataFrame from ranker.py and generates a clean,
self-contained HTML report that a recruiter can open in any browser.
No server needed - everything is embedded in one file.

"""

import os
import pandas as pd


#Tier colours (matched with ranker.py tier system)
TIER_COLORS = {
    "Strong Match"  : "#22c55e",   # green
    "Good Match"    : "#3b82f6",   # blue
    "Partial Match" : "#f59e0b",   # amber
    "Weak Match"    : "#ef4444",   # red
}

TIER_BG = {
    "Strong Match"  : "#f0fdf4",
    "Good Match"    : "#eff6ff",
    "Partial Match" : "#fffbeb",
    "Weak Match"    : "#fef2f2",
}


def _tier_badge(tier: str) -> str:
    """Return an HTML badge span for a tier label."""
    color = TIER_COLORS.get(tier, "#6b7280")
    bg    = TIER_BG.get(tier, "#f3f4f6")
    return (
        f'<span style="background:{bg}; color:{color}; border:1px solid {color}; '
        f'padding:2px 10px; border-radius:20px; font-size:11px; '
        f'font-weight:600; letter-spacing:0.5px;">{tier}</span>'
    )


def _score_bar(score: float, color: str = "#3b82f6") -> str:
    """Return an HTML progress bar for a score value (0-100)."""
    width = min(max(score, 0), 100)
    return (
        f'<div style="background:#e5e7eb; border-radius:4px; height:8px; width:100%; min-width:80px;">'
        f'<div style="background:{color}; width:{width}%; height:8px; border-radius:4px; '
        f'transition:width 0.3s;"></div></div>'
        f'<span style="font-size:11px; color:#6b7280; margin-top:2px; display:block;">{score}%</span>'
    )


def _skill_tags(skills_str: str, color: str = "#3b82f6", bg: str = "#eff6ff") -> str:
    """Return HTML skill tag spans from a comma-separated skills string."""
    if not skills_str or skills_str.strip() == "None":
        return '<span style="color:#9ca3af; font-size:12px;">—</span>'
    tags = []
    for skill in skills_str.split(","):
        skill = skill.strip()
        if skill:
            tags.append(
                f'<span style="background:{bg}; color:{color}; border:1px solid {color}33; '
                f'padding:2px 8px; border-radius:4px; font-size:11px; '
                f'margin:2px; display:inline-block;">{skill}</span>'
            )
    return " ".join(tags)


# ── Score distribution summary cards ─────────────────────────────────────────

def _summary_cards(ranked_df: pd.DataFrame, jd: dict) -> str:
    """Build the top summary stat cards section."""
    total     = len(ranked_df)
    top_score = ranked_df["Final_Score"].max()
    avg_score = round(ranked_df["Final_Score"].mean(), 1)
    strong    = (ranked_df["Final_Score"] >= 40).sum()
    good      = ((ranked_df["Final_Score"] >= 25) & (ranked_df["Final_Score"] < 40)).sum()

    cards = [
        ("Total Candidates", str(total),       "#3b82f6", "#eff6ff"),
        ("Top Score",        f"{top_score}%",  "#22c55e", "#f0fdf4"),
        ("Average Score",    f"{avg_score}%",  "#f59e0b", "#fffbeb"),
        ("Strong Matches",   str(strong),      "#22c55e", "#f0fdf4"),
        ("Good Matches",     str(good),        "#3b82f6", "#eff6ff"),
    ]

    html = '<div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:32px;">'
    for label, value, color, bg in cards:
        html += f"""
        <div style="background:{bg}; border:1px solid {color}33; border-radius:12px;
                    padding:20px 28px; flex:1; min-width:120px; text-align:center;">
            <div style="font-size:28px; font-weight:700; color:{color};">{value}</div>
            <div style="font-size:12px; color:#6b7280; margin-top:4px;">{label}</div>
        </div>"""
    html += "</div>"
    return html


# ── Candidates table ──────────────────────────────────────────────────────────

def _candidates_table(ranked_df: pd.DataFrame) -> str:
    """Build the main candidates ranking table."""

    # Add tier column if not present
    if "Tier" not in ranked_df.columns:
        from ranker import assign_tier
        ranked_df = ranked_df.copy()
        ranked_df["Tier"] = ranked_df["Final_Score"].apply(assign_tier)

    header = """
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
      <thead>
        <tr style="background:#f8fafc; border-bottom:2px solid #e2e8f0;">
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600;">Rank</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600;">Candidate ID</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600;">Tier</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600; min-width:120px;">Final Score</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600; min-width:120px;">Text Match</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600; min-width:120px;">Skill Match</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600;">Matched Skills</th>
          <th style="padding:12px 16px; text-align:left; color:#374151; font-weight:600;">Missing Skills</th>
        </tr>
      </thead>
      <tbody>"""

    rows = ""
    for _, row in ranked_df.iterrows():
        tier      = row.get("Tier", "Weak Match")
        bg        = "#ffffff" if int(row["Rank"]) % 2 == 0 else "#fafafa"
        fin_color = TIER_COLORS.get(tier, "#6b7280")

        rows += f"""
        <tr style="background:{bg}; border-bottom:1px solid #f1f5f9;
                   transition:background 0.15s;" onmouseover="this.style.background='#f0f9ff'"
                   onmouseout="this.style.background='{bg}'">
          <td style="padding:14px 16px; font-weight:700; color:#374151;">#{int(row['Rank'])}</td>
          <td style="padding:14px 16px; font-family:monospace; color:#374151;">{row['Candidate_ID']}</td>
          <td style="padding:14px 16px;">{_tier_badge(tier)}</td>
          <td style="padding:14px 16px;">{_score_bar(row['Final_Score'], fin_color)}</td>
          <td style="padding:14px 16px;">{_score_bar(row['Cosine_Score'], '#6366f1')}</td>
          <td style="padding:14px 16px;">{_score_bar(row['Skill_Match'], '#22c55e')}</td>
          <td style="padding:14px 16px;">{_skill_tags(row['Matched_Skills'], '#16a34a', '#f0fdf4')}</td>
          <td style="padding:14px 16px;">{_skill_tags(row['Missing_Skills'], '#dc2626', '#fef2f2')}</td>
        </tr>"""

    footer = "</tbody></table>"
    return header + rows + footer


# ── Full HTML page ────────────────────────────────────────────────────────────

def generate_html_report(ranked_df: pd.DataFrame, jd: dict) -> str:
    """
    Generate a complete self-contained HTML report string.

    Args:
        ranked_df : Full ranked DataFrame (output of build_ranked_results).
        jd        : Job description dict (output of load_job_description).

    Returns:
        HTML string ready to be written to a .html file.
    """
    summary_cards     = _summary_cards(ranked_df, jd)
    candidates_table  = _candidates_table(ranked_df)
    top_candidate     = ranked_df.iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Resume Screening Report – {jd['title']}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f1f5f9;
      color: #1e293b;
      padding: 32px;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .header {{
      background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
      color: white;
      border-radius: 16px;
      padding: 36px 40px;
      margin-bottom: 28px;
    }}
    .header h1 {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .header p {{
      color: #94a3b8;
      font-size: 14px;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 24px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .card h2 {{
      font-size: 15px;
      font-weight: 600;
      color: #374151;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid #f1f5f9;
    }}
    .best-candidate {{
      background: linear-gradient(135deg, #f0fdf4, #dcfce7);
      border: 1px solid #86efac;
      border-radius: 12px;
      padding: 24px 32px;
      margin-bottom: 24px;
    }}
    .best-candidate h2 {{
      color: #15803d;
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 8px;
    }}
    .best-candidate .name {{
      font-size: 22px;
      font-weight: 700;
      color: #14532d;
      margin-bottom: 4px;
    }}
    .table-wrapper {{
      overflow-x: auto;
    }}
    .footer {{
      text-align: center;
      color: #94a3b8;
      font-size: 12px;
      margin-top: 32px;
      padding-top: 20px;
      border-top: 1px solid #e2e8f0;
    }}
  </style>
</head>
<body>
  <div class="container">

    <!-- Header -->
    <div class="header">
      <h1>Resume Screening Report</h1>
      <p>Role: <strong style="color:white;">{jd['title']}</strong>
         &nbsp;·&nbsp; {len(ranked_df)} candidates screened
         &nbsp;·&nbsp; Ranked by TF-IDF similarity + skill match</p>
    </div>

    <!-- Best candidate -->
    <div class="best-candidate">
      <h2>🏆 Top Candidate</h2>
      <div class="name">Candidate {top_candidate['Candidate_ID']}</div>
      <p style="color:#166534; font-size:13px; margin-top:4px;">
        Final Score: <strong>{top_candidate['Final_Score']}%</strong>
        &nbsp;·&nbsp; Matched: {top_candidate['Matched_Skills']}
      </p>
    </div>

    <!-- Summary cards -->
    <div class="card">
      <h2>Overview</h2>
      {summary_cards}
    </div>

    <!-- Full ranking table -->
    <div class="card">
      <h2>Full Candidate Rankings</h2>
      <div class="table-wrapper">
        {candidates_table}
      </div>
    </div>

    <!-- Score legend -->
    <div class="card">
      <h2>Score Legend</h2>
      <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:13px;">
        <div><strong>Final Score</strong> — weighted average of text similarity (50%) and skill match (50%)</div>
        <div><strong>Text Match</strong> — TF-IDF cosine similarity between resume and job description</div>
        <div><strong>Skill Match</strong> — % of required JD skills found in the resume</div>
      </div>
      <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:16px;">
        {''.join([f'<span>{_tier_badge(t)} &nbsp; {t} = {v}%+</span>' for t, v in
                  [("Strong Match","40"), ("Good Match","25"),
                   ("Partial Match","10"), ("Weak Match","0")]])}
      </div>
    </div>

    <div class="footer">
      Generated by Resume Screening System &nbsp;·&nbsp; Role: {jd['title']}
    </div>

  </div>
</body>
</html>"""
    return html


# ── Save report ───────────────────────────────────────────────────────────────

def save_html_report(
    ranked_df  : pd.DataFrame,
    jd         : dict,
    output_dir : str = None,
) -> str:
    """
    Generate and save the HTML report to disk.

    Args:
        ranked_df  : Ranked DataFrame from build_ranked_results().
        jd         : Job description dict.
        output_dir : Folder to save into (defaults to ../results).

    Returns:
        Path to the saved HTML file.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results"
        )

    os.makedirs(output_dir, exist_ok=True)

    safe_title = jd["title"].replace(" ", "_").replace("/", "-").lower()
    filepath   = os.path.join(output_dir, f"report_{safe_title}.html")

    html = generate_html_report(ranked_df, jd)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report] HTML report saved → {filepath}")
    return filepath


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from preprocess import load_resumes, load_job_description
    from extractor  import (
        load_skills_dict, build_matcher,
        extract_skills_from_jd, extract_skills_from_dataframe,
        compute_skill_gap,
    )
    from scorer import (
        compute_cosine_scores, compute_final_scores, build_ranked_results,
    )
    from ranker import add_tier_column

    RESUME_PATH = "data/Resume/Resume.csv"
    JD_PATH     = "data/job_descriptions.csv"

    # Run full pipeline
    resumes_df    = load_resumes(RESUME_PATH, category_filter="ENGINEERING")
    jd            = load_job_description(JD_PATH, role_filter="Software Engineer")
    skills        = load_skills_dict()
    matcher       = build_matcher(skills)
    jd_skills     = extract_skills_from_jd(jd, matcher)
    resume_skills = extract_skills_from_dataframe(resumes_df, matcher)
    skill_gaps    = [compute_skill_gap(jd_skills, resume_skills[i]) for i in range(len(resumes_df))]
    cosine_scores = compute_cosine_scores(resumes_df, jd)
    final_scores  = compute_final_scores(cosine_scores, skill_gaps)
    ranked_df     = build_ranked_results(resumes_df, jd, cosine_scores, skill_gaps, final_scores)
    ranked_df     = add_tier_column(ranked_df)

    # Generate and save HTML report
    path = save_html_report(ranked_df, jd)
    print(f"\nOpen this file in your browser:\n  {os.path.abspath(path)}")