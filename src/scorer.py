"""
scorer.py
Scoring and ranking pipeline for the Resume Screening System.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#TF-IDF Vectorizer

def build_tfidf_vectorizer(corpus: list[str]) -> tuple:
    #Fit a TF-IDF vectorizer on a corpus of documents.
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), # capture unigrams and bigrams (e.g. 'machine learning')
        min_df=1, # include terms that appear in at least 1 doc
        max_df=0.95,# ignore terms appearing in 95%+ of docs (too common)
        sublinear_tf=True, # apply log normalization to term frequency
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


#Cosine Similarity Scoring 

def compute_cosine_scores(resumes_df: pd.DataFrame, jd: dict) -> list[float]:
    """
    Compute cosine similarity between each resume and the job description.
    Uses clean_text from both resumes and JD (output of preprocess.py).
    JD is always the first document in the corpus so its vector is index 0.
    """
    jd_text      = jd["clean_text"]
    resume_texts = resumes_df["clean_text"].tolist()

    # Build corpus: JD first, then all resumes
    corpus = [jd_text] + resume_texts

    _, matrix = build_tfidf_vectorizer(corpus)

    # JD vector is row 0; resume vectors are rows 1 onwards
    jd_vector      = matrix[0]
    resume_vectors = matrix[1:]

    # Compute similarity of each resume against the JD
    scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    print(f"[scorer] Cosine similarity computed for {len(scores)} resumes.")
    return scores.tolist()


#Combined Scoring

def compute_final_scores(
    cosine_scores: list[float],
    skill_gaps: list[dict],
    cosine_weight: float = 0.5,
    skill_weight: float  = 0.5,
) -> list[float]:
    """
    Combine cosine similarity score and skill match score into one final score.

    Formula:
        final_score = (cosine_score * cosine_weight)+ (skill_match_pct/100 * skill_weight)

    Both weights default to 0.5 (equal importance).
    Adjust weights to prioritise text similarity or skill coverage.
    """
    assert abs(cosine_weight + skill_weight - 1.0) < 1e-6, \
        "cosine_weight + skill_weight must equal 1.0"

    final_scores = []
    for cos_score, gap in zip(cosine_scores, skill_gaps):
        skill_score  = gap["match_pct"] / 100.0
        final        = (cos_score * cosine_weight) + (skill_score * skill_weight)
        final_scores.append(round(final, 4))

    return final_scores


#Ranked Results Builder 

def build_ranked_results(
    resumes_df     : pd.DataFrame,
    jd             : dict,
    cosine_scores  : list[float],
    skill_gaps     : list[dict],
    final_scores   : list[float],
) -> pd.DataFrame:
   
    #Combine all scores and skill gap data into a single ranked DataFrame.
    records = []
    for i, (_, row) in enumerate(resumes_df.iterrows()):
        gap = skill_gaps[i]
        records.append({
            "Candidate_ID": row["ID"],
            "Final_Score" : round(final_scores[i] * 100, 2),
            "Cosine_Score": round(cosine_scores[i] * 100, 2),
            "Skill_Match": gap["match_pct"],
            "Matched_Skills": ", ".join(gap["matched"]) if gap["matched"] else "None",
            "Missing_Skills": ", ".join(gap["missing"]) if gap["missing"] else "None",
            "Extra_Skills" : ", ".join(gap["extra"])   if gap["extra"]   else "None",
        })

    df = pd.DataFrame(records)

    # Sort by Final_Score descending
    df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

    # Add rank column starting at 1
    df.insert(0, "Rank", df.index + 1)

    print(f"[scorer] Ranking complete. Top candidate score: {df.iloc[0]['Final_Score']}%")
    return df


#Console Report 

def print_ranking_report(ranked_df: pd.DataFrame, jd: dict, top_n: int = 10) -> None:
   # Print a clean, human-readable ranking report to the console.
   
    print("\n")
    print(f"RESUME SCREENING REPORT")
    print(f"Role : {jd['title']}")
    print(f"Candidates : {len(ranked_df)}")
    print(f"Showing: Top {min(top_n, len(ranked_df))}")
    

    for _, row in ranked_df.head(top_n).iterrows():
        print(f"\n  Rank #{int(row['Rank'])}  |  Candidate {row['Candidate_ID']}")
        print(f"  {'─'*55}")
        print(f"  Final Score    : {row['Final_Score']}%")
        print(f"  Cosine Score   : {row['Cosine_Score']}%")
        print(f"  Skill Match    : {row['Skill_Match']}%")
        print(f"  Matched Skills : {row['Matched_Skills']}")
        print(f"  Missing Skills : {row['Missing_Skills']}")
        print(f"  Extra Skills   : {row['Extra_Skills']}")

    print(f"\n{'='*65}\n")


#Score Distribution Summary

def print_score_distribution(ranked_df: pd.DataFrame) -> None:
    """
    Print a summary of how scores are distributed across all candidates.
    Useful for understanding the spread and setting shortlist thresholds.
    """
    scores = ranked_df["Final_Score"]
    print("\n")
    print(f"  SCORE DISTRIBUTION")
    print(f"  Total candidates: {len(scores)}")
    print(f"  Highest score : {scores.max()}%")
    print(f"  Lowest score: {scores.min()}%")
    print(f"  Average score: {round(scores.mean(), 2)}%")
    print(f"  Median score: {round(scores.median(), 2)}%")
    print(f"  {'─'*35}")
    print(f"  Above 50% : {(scores >= 50).sum()} candidates")
    print(f"  Above 30% : {(scores >= 30).sum()} candidates")
    print(f"  Above 10% : {(scores >= 10).sum()} candidates")
    print(f"  Below 10%: {(scores < 10).sum()} candidates")


#Shortlist

def get_shortlist(ranked_df: pd.DataFrame, threshold: float = 20.0) -> pd.DataFrame:
    """
    Return only candidates above a minimum score threshold.
    """
    shortlist = ranked_df[ranked_df["Final_Score"] >= threshold].copy()
    print(f"[scorer] Shortlisted {len(shortlist)} candidates above {threshold}% threshold.")
    return shortlist


#Quick test 
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from preprocess import load_resumes, load_job_description
    from extractor  import (
        load_skills_dict, build_matcher,
        extract_skills_from_jd, extract_skills_from_dataframe,
        compute_skill_gap,
    )

    RESUME_PATH = "data/Resume/Resume.csv"
    JD_PATH = "data/job_descriptions.csv"

    # 1. Load data
    resumes_df = load_resumes(RESUME_PATH, category_filter="ENGINEERING")
    jd = load_job_description(JD_PATH, role_filter="Software Engineer")

    # 2. Extract skills
    skills = load_skills_dict()
    matcher = build_matcher(skills)
    jd_skills = extract_skills_from_jd(jd, matcher)
    resume_skills = extract_skills_from_dataframe(resumes_df, matcher)

    # 3. Compute skill gaps
    skill_gaps = [
        compute_skill_gap(jd_skills, resume_skills[i])
        for i in range(len(resumes_df))
    ]

    # 4. Compute cosine similarity scores
    cosine_scores = compute_cosine_scores(resumes_df, jd)

    # 5. Compute final combined scores
    final_scores = compute_final_scores(cosine_scores, skill_gaps)

    # 6. Build ranked results DataFrame
    ranked_df = build_ranked_results(
        resumes_df, jd, cosine_scores, skill_gaps, final_scores
    )

    # 7. Print reports
    print_ranking_report(ranked_df, jd, top_n=10)
    print_score_distribution(ranked_df)

    # 8. Shortlist candidates above 20%
    shortlist = get_shortlist(ranked_df, threshold=20.0)
    if not shortlist.empty:
        print(f"\nShortlisted Candidates:\n{shortlist[['Rank','Candidate_ID','Final_Score','Matched_Skills']].to_string(index=False)}")