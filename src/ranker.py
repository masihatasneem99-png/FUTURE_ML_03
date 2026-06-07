"""
ranker.py
---------
Final ranking and reporting module for the Resume Screening System.

This is the last step in the pipeline. It takes the scored and ranked
DataFrame from scorer.py and produces clean, actionable output for
recruiters - showing who the top candidates are, what skills they have,
and what they're missing.

I kept this module focused on presentation and decision-support.
The actual math happens in scorer.py - ranker.py just makes the
results readable and usable.
"""

import os
import pandas as pd


TIERS = {
    "Strong Match"  : 40.0,
    "Good Match"    : 25.0,
    "Partial Match" : 10.0,
    "Weak Match"    : 0.0,
}


def assign_tier(score: float) -> str:
    """
    Assigning a human-readable tier label based on final score.

    Tiers:
      Strong Match  - 40% and above
      Good Match    - 25% to 39%
      Partial Match - 10% to 24%
      Weak Match    - below 10%
    """
    for label, threshold in TIERS.items():
        if score >= threshold:
            return label
    return "Weak Match"


def add_tier_column(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """Add a Tier column to the ranked DataFrame based on Final_Score."""
    df = ranked_df.copy()
    df["Tier"] = df["Final_Score"].apply(assign_tier)
    return df


# ── Top N candidates ──────────────────────────────────────────────────────────

def get_top_candidates(ranked_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns the top N candidates from the ranked list.
    If fewer than N candidates exist, returns all of them.

    Args:
        ranked_df : Full ranked DataFrame from scorer.py
        n         : Number of candidates to return (default 10)

    Returns:
        DataFrame of top N candidates with tier labels added.
    """
    df = add_tier_column(ranked_df)
    top = df.head(n).copy()
    print(f"[ranker] Returning top {len(top)} candidates out of {len(df)} total.")
    return top


#Console summary

def print_final_report(ranked_df: pd.DataFrame, jd: dict, top_n: int = 10) -> None:
    """
    Print the final ranking report to the console.

    Shows each candidate's rank, tier, scores, and skill summary.
    Written to be readable by a recruiter, not just a developer.
    """
    df = add_tier_column(ranked_df)
    top = df.head(top_n)

    print(f"\n")
    print(f"CANDIDATE RANKING REPORT")
    print(f"Role: {jd['title']}")
    print(f"Total CVs  : {len(ranked_df)}")
    print(f"Top {top_n} shown below")
    

    for _, row in top.iterrows():
        print(f"""
  #{int(row['Rank'])} {row['Tier'].upper()}
  Candidate ID   : {row['Candidate_ID']}
  Final Score    : {row['Final_Score']}%
  Text Similarity: {row['Cosine_Score']}%
  Skill Match    : {row['Skill_Match']}%
  Matched Skills : {row['Matched_Skills']}
  Missing Skills : {row['Missing_Skills']}
  {'─'*55}""")

    print()


#Tier breakdown

def print_tier_summary(ranked_df: pd.DataFrame) -> None:
    """
    Show how many candidates fall into each tier.
    Good for giving a recruiter a quick sense of the talent pool.
    """
    df = add_tier_column(ranked_df)
    counts = df["Tier"].value_counts()

    print(f"\n")
    print(f"TALENT POOL SUMMARY")
    print(f"{'='*40}")
    for tier in TIERS.keys():
        count = counts.get(tier, 0)
        bar= "█" * count if count <= 40 else "█" * 40 + f"  (+{count - 40})"
        print(f"  {tier:<15} : {count:>3}  {bar}")
    


# CSV export

def export_to_csv(ranked_df, jd, output_dir=None):
    """
    Save the full ranked results to a CSV file.

    The filename includes the job title so you can run this for
    multiple roles without overwriting previous results.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    
    os.makedirs(output_dir, exist_ok=True)

    # Clean up the job title for use in a filename
    safe_title = jd["title"].replace(" ", "_").replace("/", "-").lower()
    filepath   = os.path.join(output_dir, f"ranked_{safe_title}.csv")

    df = add_tier_column(ranked_df)
    df.to_csv(filepath, index=False)

    print(f"[ranker] Results saved to {filepath}")
    return filepath


#Best candidate summary

def print_best_candidate(ranked_df: pd.DataFrame, jd: dict) -> None:
    """
    Print a one-line summary of the top candidate.
    Useful for a quick answer to 'who is the best fit?'
    """
    best = ranked_df.iloc[0]
    print(f"\n Best fit for '{jd['title']}' → Candidate {best['Candidate_ID']}")
    print(f"Score: {best['Final_Score']}% | Matched: {best['Matched_Skills']}")
    print(f"Missing: {best['Missing_Skills']}\n")


# Run the full pipeline 

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
        print_score_distribution,
    )

    RESUME_PATH = "data/Resume/Resume.csv"
    JD_PATH= "data/job_descriptions.csv"

    # Step 1: Load
    print("\n[1/6] Loading data...")
    resumes_df = load_resumes(RESUME_PATH, category_filter="ENGINEERING")
    jd= load_job_description(JD_PATH, role_filter="Software Engineer")

    # Step 2: Extract skills 
    print("\n[2/6] Extracting skills...")
    skills        = load_skills_dict()
    matcher       = build_matcher(skills)
    jd_skills     = extract_skills_from_jd(jd, matcher)
    resume_skills = extract_skills_from_dataframe(resumes_df, matcher)

    # Step 3: Skill gap analysis
    print("\n[3/6] Computing skill gaps...")
    skill_gaps = [
        compute_skill_gap(jd_skills, resume_skills[i])
        for i in range(len(resumes_df))
    ]

    # Step 4: Cosine similarity
    print("\n[4/6] Scoring resumes...")
    cosine_scores = compute_cosine_scores(resumes_df, jd)
    final_scores  = compute_final_scores(cosine_scores, skill_gaps)

    # Step 5: Rank
    print("\n[5/6] Ranking candidates...")
    ranked_df = build_ranked_results(
        resumes_df, jd, cosine_scores, skill_gaps, final_scores
    )

    # Step 6: Report
    print("\n[6/6] Generating report...")
    print_best_candidate(ranked_df, jd)
    print_tier_summary(ranked_df)
    print_final_report(ranked_df, jd, top_n=10)
    print_score_distribution(ranked_df)

    # Step 7: Save results
    # Build path relative to this file's location, not the working directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    export_to_csv(ranked_df, jd, output_dir=output_dir)