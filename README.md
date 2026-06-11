# Resume Screening System

An NLP-based resume screening tool that automatically scores, ranks, and compares candidates against a job description. Built with Python, spaCy, scikit-learn, and Streamlit.

---

## Overview

Manually reviewing hundreds of resumes is slow and inconsistent. This system automates the process by extracting skills from resumes, comparing them against a job description, and ranking candidates by relevance — giving recruiters a clear, data-driven shortlist.

---

## Features

- Upload a resume and get an instant match score against any job role
- Skill extraction using spaCy's PhraseMatcher
- TF-IDF cosine similarity scoring between resume and job description
- Skill gap analysis — matched skills, missing skills, extra skills
- Candidate tier labels — Strong, Good, Partial, Weak Match
- Full analytics dashboard with score distribution and top candidate rankings
- Export ranked results as CSV
- Clean Streamlit web interface

---

## Project Structure

```
Resume_Screening_System/
│
├── data/
│   └── Resume/
│       ├── Resume.csv               # Kaggle resume dataset
│       └── job_descriptions.csv     # Job descriptions dataset
│
├── results/
│   ├── ranked_software_engineer.csv # Ranked candidate output
│   └── report_software_engineer.html# Visual HTML report
│
├── src/
│   ├── app.py          # Streamlit web application
│   ├── preprocess.py   # Text cleaning and data loading
│   ├── extractor.py    # Skill extraction with spaCy
│   ├── scorer.py       # TF-IDF scoring and cosine similarity
│   ├── ranker.py       # Candidate ranking and tier assignment
│   └── report.py       # HTML report generator
│
├── .gitignore
├── pyvenv.cfg
└── README.md
```

---

## How It Works

**1. Preprocessing** — Raw resume text is cleaned, tokenized, stopwords removed, and lemmatized using NLTK and spaCy.

**2. Skill Extraction** — spaCy's PhraseMatcher scans each resume and the job description against a dictionary of 170+ tech skills to identify exact matches.

**3. Scoring** — Each resume is scored using two signals combined equally:
- **Text Similarity (50%)** — TF-IDF cosine similarity between resume and JD
- **Skill Match (50%)** — percentage of required JD skills found in the resume

**4. Ranking** — Candidates are sorted by final score and assigned a tier label.

```
Final Score = (Cosine Similarity × 0.5) + (Skill Match % × 0.5)
```

| Tier | Score |
|------|-------|
| Strong Match | 40%+ |
| Good Match | 25–39% |
| Partial Match | 10–24% |
| Weak Match | Below 10% |

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/resume-screening-system.git
cd resume-screening-system
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLP models**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

**5. Add datasets**

Download both datasets from Kaggle and place them in `data/Resume/`:
- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) → `Resume.csv`
- [Job Descriptions Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) → `job_descriptions.csv`

---

## Run the App

```bash
streamlit run src/app.py
```

---

## Dependencies

```
pandas
numpy
nltk
spacy
scikit-learn
pdfminer.six
matplotlib
seaborn
streamlit
jupyter
ipykernel
```

---

## Dataset

- **Resumes** — [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Sneha Anbhawal (~2,400 resumes across 25 categories)
- **Job Descriptions** — [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) by Ravindra Singh Rana

Datasets are not included in this repository due to file size.

---

## Results

Sample output for the **Software Engineer** role screened against **ENGINEERING** category resumes:

| Rank | Candidate ID | Tier | Final Score | Matched Skills |
|------|-------------|------|-------------|----------------|
| 1 | 54227873 | Good Match | 38.4% | java, python |
| 2 | 12938471 | Partial Match | 18.2% | sql, api |
| 3 | 38471920 | Partial Match | 11.6% | java |

Full results available in [`results/ranked_software_engineer.csv`](results/ranked_software_engineer.csv).

---


## Author

**Masiha Tasneem**
- GitHub: https://github.com/masihatasneem99-png
- LinkedIn: www.linkedin.com/in/masihatasneem

---
## License

This project is for educational purposes.