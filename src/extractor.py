"""
extractor.py
------------
Skill extraction pipeline for the Resume Screening System.
"""

import os
import spacy
from spacy.matcher import PhraseMatcher

#Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

#Built-in skills dictionary (IT / Tech focused)
# Used when no external skills file is provided.
DEFAULT_SKILLS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "r",
    "scala", "kotlin", "swift", "go", "golang", "rust", "php", "ruby",
    "bash", "shell", "perl", "matlab", "vba",

    # Web frameworks & libraries
    "django", "flask", "fastapi", "spring", "spring boot", "react", "angular",
    "vue", "node", "node.js", "express", "next.js", "html", "css", "bootstrap",
    "jquery", "rest", "restful", "graphql", "soap", "api",

    # Data science & ML
    "machine learning", "deep learning", "neural network", "natural language processing",
    "nlp", "computer vision", "data science", "data analysis", "data mining",
    "feature engineering", "model training", "model deployment", "transfer learning",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "xgboost",
    "lightgbm", "catboost", "hugging face", "transformers", "bert", "gpt",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "oracle", "microsoft sql server",
    "mongodb", "cassandra", "redis", "elasticsearch", "dynamodb", "firebase",
    "neo4j", "hbase", "couchdb",

    # Big data
    "hadoop", "spark", "apache spark", "kafka", "hive", "pig", "flink",
    "airflow", "luigi", "dbt", "etl", "data pipeline", "data warehouse",
    "data lake", "snowflake", "redshift", "bigquery",

    # Cloud platforms
    "aws", "amazon web services", "azure", "microsoft azure", "gcp",
    "google cloud", "heroku", "digitalocean", "lambda", "ec2", "s3",
    "rds", "cloudformation", "terraform",

    # DevOps & tools
    "docker", "kubernetes", "jenkins", "git", "github", "gitlab", "bitbucket",
    "ci/cd", "devops", "ansible", "puppet", "chef", "linux", "unix",
    "nginx", "apache", "microservices", "agile", "scrum", "jira", "confluence",

    # Data visualization & BI
    "tableau", "power bi", "looker", "qlik", "excel", "google sheets",
    "d3.js", "dash",

    # Networking & security
    "networking", "tcp/ip", "dns", "firewall", "vpn", "cybersecurity",
    "penetration testing", "ethical hacking", "ssl", "oauth",

    # Soft skills (useful for JD matching)
    "communication", "leadership", "problem solving", "teamwork",
    "project management", "time management", "critical thinking",
    "collaboration", "analytical", "presentation",
]


#Skills dictionary loader

def load_skills_dict(filepath: str = None) -> list[str]:
    """
    Load skills from an external text file (one skill per line).
    Falls back to DEFAULT_SKILLS if no file is provided or file not found.
    """
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            skills = [line.strip().lower() for line in f if line.strip()]
        print(f"[extractor] Loaded {len(skills)} skills from {filepath}")
        return skills

    print(f"[extractor] Using built-in skills dictionary ({len(DEFAULT_SKILLS)} skills).")
    return [s.lower() for s in DEFAULT_SKILLS]


#PhraseMatcher builder 
def build_matcher(skills: list[str]) -> PhraseMatcher:
    """
    Build a spaCy PhraseMatcher from a list of skill strings.
    Uses lowercase matching for robustness.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILLS", patterns)
    return matcher


#Core extraction function 

def extract_skills(text: str, matcher: PhraseMatcher) -> set[str]:
    #Extract skills found in a text string using the PhraseMatcher.
  
    if not isinstance(text, str) or not text.strip():
        return set()

    doc = nlp(text.lower())
    matches = matcher(doc)

    found_skills = set()
    for _, start, end in matches:
        skill = doc[start:end].text.strip().lower()
        found_skills.add(skill)

    return found_skills


#Resume-level extraction

def extract_skills_from_resume(resume_text: str, matcher: PhraseMatcher) -> set[str]:
    """
    Extract skills from a single resume's raw text.
    Runs on Resume_str (original text) for best recall —
    cleaned text may have stripped some multi-word skill names.
    """
    return extract_skills(resume_text, matcher)


def extract_skills_from_dataframe(df, matcher: PhraseMatcher) -> list[set]:
    #Extract skills from every resume in the DataFrame.

    print(f"[extractor] Extracting skills from {len(df)} resumes...")
    skills_list = [
        extract_skills_from_resume(row["Resume_str"], matcher)
        for _, row in df.iterrows()
    ]
    print(f"[extractor] Done.")
    return skills_list


# JD-level extraction

def extract_skills_from_jd(jd: dict, matcher: PhraseMatcher) -> set[str]:
    # Use the 'skills' column text directly — it's already a clean skill list
    skills_text = jd.get("skills_raw", jd["raw_text"])
    required = extract_skills(skills_text, matcher)
    print(f"[extractor] JD '{jd['title']}' requires {len(required)} skills: {sorted(required)}")
    return required

#Skill gap analysis

def compute_skill_gap(required_skills: set, candidate_skills: set) -> dict:
    
    #Compare a candidate's skills against what the JD requires.
    
    matched = required_skills & candidate_skills
    missing = required_skills - candidate_skills
    extra = candidate_skills - required_skills

    match_pct = (
        round(len(matched) / len(required_skills) * 100, 1)
        if required_skills else 0.0
    )

    return {
        "matched":sorted(matched),
        "missing":sorted(missing),
        "extra": sorted(extra),
        "match_pct": match_pct,
    }    


#Utility: summary printer

def print_skill_summary(candidate_id, gap: dict) -> None:
    """Print a readable skill gap summary for one candidate."""
    print(f"\n{'─'*55}")
    print(f"Candidate ID: {candidate_id}")
    print(f"Skill Match: {gap['match_pct']}%")
    print(f"Matched Skills: {', '.join(gap['matched']) if gap['matched'] else 'None'}")
    print(f"Missing Skills: {', '.join(gap['missing']) if gap['missing'] else 'None'}")
    print(f"Extra Skills: {', '.join(gap['extra'][:5]) if gap['extra'] else 'None'}")
    print(f"{'─'*55}")


#Quick test

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_resumes, load_job_description

    RESUME_PATH = "data/Resume/Resume.csv"
    JD_PATH     = "data/job_descriptions.csv"

    #Load data
    resumes_df = load_resumes(RESUME_PATH, category_filter="ENGINEERING")
    jd         = load_job_description(JD_PATH, role_filter="Software Engineer")

    #  Build matcher
    skills  = load_skills_dict()
    matcher = build_matcher(skills)

    # Extract skills
    jd_skills     = extract_skills_from_jd(jd, matcher)
    resume_skills = extract_skills_from_dataframe(resumes_df, matcher)

    # Score and SORT by match %
    scored = []
    for i, (_, row) in enumerate(resumes_df.iterrows()):
        gap = compute_skill_gap(jd_skills, resume_skills[i])
        scored.append((row["ID"], gap))

    scored.sort(key=lambda x: x[1]["match_pct"], reverse=True)

    # Print TOP 5 only
    print("\n===== TOP 5 CANDIDATES BY SKILL MATCH =====")
    for candidate_id, gap in scored[:5]:
        print_skill_summary(candidate_id, gap)