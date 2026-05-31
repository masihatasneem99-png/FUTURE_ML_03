"""
preprocess.py:
Text cleaning and preprocessing pipeline for the Resume Screening System.
"""

import re
import string
from numpy.char import title
import pandas as pd
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#Download required NLTK data (safe to run multiple times)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

#Loading spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

#Constants
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Extra domain-specific stopwords that add noise in resume context
EXTRA_STOPWORDS = {
    "experience", "work", "working", "worked", "responsibilities",
    "responsible", "role", "position", "skills", "ability", "strong",
    "excellent", "good", "knowledge", "understanding", "team", "using",
    "used", "use", "years", "year", "etc", "also", "well", "including",
    "various", "within", "across", "per", "via", "e.g", "i.e",
}

ALL_STOPWORDS = STOP_WORDS | EXTRA_STOPWORDS


#Core text cleaning

def remove_urls(text: str) -> str:
    """Remove http/https URLs and www links."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r"\S+@\S+", " ", text)


def remove_phone_numbers(text: str) -> str:
    """Remove common phone number formats."""
    return re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Keep only letters, numbers, and single spaces.
    Preserves '+' in tech terms like 'C++' before this step
    but removes all other punctuation after lowercasing.
    """
    # Replacing newlines, tabs with space
    text = re.sub(r"[\n\r\t]", " ", text)
    # Removes bullet points and special unicode characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Removes punctuation except hyphens (useful for skill names like 'sci-kit')
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
       Lowercase; Remove URLs, emails, phone numbers; Remove special characters
       Returns clean but un-tokenized string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)
    return text


#Tokenization and normalization

def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Tokenize cleaned text, remove stopwords, and lemmatize.
    Returns a list of meaningful tokens.
    """
    tokens = word_tokenize(text)
    processed = []
    for token in tokens:
        # Skipping stopwords and single-character tokens
        if token in ALL_STOPWORDS or len(token) < 2:
            continue
        # Skipping pure numbers
        if token.isdigit():
            continue
        lemma = LEMMATIZER.lemmatize(token)
        processed.append(lemma)
    return processed


def preprocess_text(text: str) -> str:
    """
    Full pipeline: clean → tokenize → lemmatize → rejoin.
    Returns a single clean string ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


#spaCy-based preprocessing (richer, used for skill extraction prep)

def spacy_preprocess(text: str) -> str:
    """
    Uses spaCy for lemmatization instead of NLTK.
    More accurate for technical terms; slightly slower.
    Use this when feeding text into the skill extractor.
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 1
        and not token.like_num
    ]
    return " ".join(tokens)


# Dataset loaders 

def load_resumes(filepath: str, category_filter: str = None) -> pd.DataFrame:
    """
    Loads and preprocess Resume.csv.

    Returns:
        DataFrame with columns:
           ID : row index
           Category: job category label
           Resume_str: original raw text
           clean_text: fully preprocessed text (for vectorization)
    """
    df = pd.read_csv(filepath)

    # Standardizing column names (handle different CSV formats)
    df.columns = [col.strip() for col in df.columns]

    # Ensure required columns exist
    required = {"Category", "Resume_str"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Resume.csv is missing columns: {missing}")

    # Optional category filter
    if category_filter:
        df = df[df["Category"].str.lower() == category_filter.lower()].copy()
        if df.empty:
            raise ValueError(
                f"No resumes found for category: '{category_filter}'. "
                f"Available: {df['Category'].unique().tolist()}"
            )

    # Drop rows with empty resume text
    df = df.dropna(subset=["Resume_str"])
    df = df[df["Resume_str"].str.strip() != ""]

    # Adding preprocessed text column
    print(f"[preprocess] Cleaning {len(df)} resumes...")
    df["clean_text"] = df["Resume_str"].apply(preprocess_text)

    # Reset index and add ID
    df = df.reset_index(drop=True)
    if "ID" not in df.columns:
        df.insert(0, "ID", df.index + 1)

    print(f"[preprocess] Done. {len(df)} resumes loaded.")
    return df[["ID", "Category", "Resume_str", "clean_text"]]


def load_job_description(filepath: str, role_filter: str = None) -> dict:
    """
    Load and preprocess job_descriptions.csv.

    Returns:
        Dict with keys:
          title : job title
          raw_text : original JD text
          clean_text: preprocessed JD text (for vectorization)
    """
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]

    # Detect text column (handles different column naming conventions)
    text_col = None
    for candidate in ["Job Description", "job_description", "description", "Description", "text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"Cannot find a description column in job_descriptions.csv. "
            f"Columns found: {df.columns.tolist()}"
        )

    # Detects title column
    title_col = None
    for candidate in ["Job Title", "job_title", "title", "Title", "Position", "Role"]:
        if candidate in df.columns:
            title_col = candidate
            break

    # Filters by role if provided
    if role_filter and title_col:
        mask = df[title_col].str.lower().str.contains(role_filter.lower(), na=False)
        matched = df[mask]
        if not matched.empty:
            df = matched

    row = df.iloc[0]
    raw_text = str(row[text_col])
    title = str(row[title_col]) if title_col else "Unknown Role"

    print(f"[preprocess] Job Description loaded: '{title}'")
    # Add this after detecting text_col
    skills_col = "skills" if "skills" in df.columns else None

    # Update the return dict
    return {
        "title":      title,
        "raw_text":   raw_text,
        "clean_text": preprocess_text(raw_text),
        "skills_raw": str(row[skills_col]) if skills_col else raw_text,
    }


#Utility

def show_sample(df: pd.DataFrame, n: int = 3) -> None:
    """Prints a quick preview of preprocessed resumes."""
    print(f"\n{'='*60}")
    print(f"SAMPLE PREPROCESSED RESUMES (showing {n})")
    print(f"{'='*60}")
    for _, row in df.head(n).iterrows():
        print(f"\n[ID {row['ID']}] Category: {row['Category']}")
        print(f"Original(first 120 chars): {row['Resume_str'][:120].strip()}...")
        print(f"Cleaned(first 120 chars): {row['clean_text'][:120].strip()}...")
    print(f"{'='*60}\n")


#Quick test

if __name__ == "__main__":
    import os

    RESUME_PATH = "data/Resume/Resume.csv"
    JD_PATH     = "data/job_descriptions.csv"

    # Verify files exist
    for path in [RESUME_PATH, JD_PATH]:
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            print("Update the path above to match your project structure.")
            exit(1)

    # Load resumes — filter to Data Science category
    resumes_df = load_resumes(RESUME_PATH, category_filter="INFORMATION-TECHNOLOGY")
    show_sample(resumes_df, n=3)

    # Load job description — search for Data Scientist role
    jd = load_job_description(JD_PATH, role_filter="Data Scientist")
    print(f"JD Title   : {jd['title']}")
    print(f"JD Raw     : {jd['raw_text'][:150]}...")
    print(f"JD Cleaned : {jd['clean_text'][:150]}...")