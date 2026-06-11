"""
app.py
------
Streamlit web application for the Resume Screening System.

Page 1 - Resume Screener : Upload a resume, pick a job role, get instant score
Page 2 - Analytics Dashboard : Full candidate rankings, charts, skill gap analysis

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import tempfile

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Adding src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_resumes, load_job_description, preprocess_text
from extractor  import (
    load_skills_dict, build_matcher,
    extract_skills_from_jd, extract_skills_from_dataframe,
    extract_skills_from_resume, compute_skill_gap,
)
from scorer import (
    compute_cosine_scores, compute_final_scores, build_ranked_results,
)
from ranker import assign_tier, add_tier_column


#Paths 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUME_CSV = os.path.join(ROOT, "data", "Resume", "Resume.csv")
JD_CSV= os.path.join(ROOT, "data", "job_descriptions.csv")

AVAILABLE_ROLES = [
    "Software Engineer",
    "Data Analyst",
    "Network Engineer",
    "Java Developer",
    "Front-End Developer",
    "IT Support Specialist",
    "Network Security Specialist",
    "Software Tester",
    "Data Engineer",
    "Network Administrator",
]

AVAILABLE_CATEGORIES = [
    "ENGINEERING",
    "INFORMATION-TECHNOLOGY",
    "DATA SCIENCE",
    "FINANCE",
    "HEALTHCARE",
]

TIER_ST_COLORS = {
    "Strong Match"  : "#22c55e",
    "Good Match"    : "#3b82f6",
    "Partial Match" : "#f59e0b",
    "Weak Match"    : "#ef4444",
}


#Page config 

st.set_page_config(
    page_title  = "Resume Screening System",
    page_icon   = "📋",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# Global styles 

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }

    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }

    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
    }

    .stButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2563eb !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }

    [data-testid="stDataFrame"] { border-radius: 8px; }

    .skill-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Cached pipeline functions

@st.cache_resource(show_spinner=False)
def get_matcher():
    skills  = load_skills_dict()
    matcher = build_matcher(skills)
    return matcher


@st.cache_data(show_spinner=False)
def load_and_score_all(category: str, role: str):
    """Run the full pipeline for the analytics dashboard."""
    resumes_df = load_resumes(RESUME_CSV, category_filter=category)
    jd = load_job_description(JD_CSV, role_filter=role)
    matcher= get_matcher()
    jd_skills = extract_skills_from_jd(jd, matcher)
    resume_skills = extract_skills_from_dataframe(resumes_df, matcher)
    skill_gaps= [
        compute_skill_gap(jd_skills, resume_skills[i])
        for i in range(len(resumes_df))
    ]
    cosine_scores = compute_cosine_scores(resumes_df, jd)
    final_scores= compute_final_scores(cosine_scores, skill_gaps)
    ranked_df = build_ranked_results(resumes_df, jd, cosine_scores, skill_gaps, final_scores)
    ranked_df = add_tier_column(ranked_df)
    return ranked_df, jd, jd_skills


def score_single_resume(resume_text: str, role: str) -> dict:
    """Score a single uploaded resume against a chosen JD."""
    jd = load_job_description(JD_CSV, role_filter=role)
    matcher = get_matcher()

    jd_skills  = extract_skills_from_jd(jd, matcher)
    candidate_skills = extract_skills_from_resume(resume_text, matcher)
    gap= compute_skill_gap(jd_skills, candidate_skills)

    # Cosine score — compare cleaned texts
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    jd_clean = jd["clean_text"]
    resume_clean = preprocess_text(resume_text)

    tfidf= TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = tfidf.fit_transform([jd_clean, resume_clean])
    cos= cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    skill_score  = gap["match_pct"] / 100.0
    final= round((cos * 0.5 + skill_score * 0.5) * 100, 2)
    tier = assign_tier(final)

    return {
        "final_score" : final,
        "cosine_score" : round(cos * 100, 2),
        "skill_match" : gap["match_pct"],
        "matched_skills" : gap["matched"],
        "missing_skills" : gap["missing"],
        "extra_skills" : gap["extra"],
        "tier" : tier,
        "jd_title" : jd["title"],
        "jd_skills" : sorted(jd_skills),
    }


# Sidebar navigation

with st.sidebar:
    st.markdown("### 📋 Resume Screening")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍  Resume Screener", "📊  Analytics Dashboard"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:12px; color:#94a3b8;'>"
        "Powered by TF-IDF + spaCy NLP<br>"
        "ENGINEERING category · Software roles</p>",
        unsafe_allow_html=True,
    )



# PAGE 1 — RESUME SCREENER

if "Resume Screener" in page:

    st.markdown("## 🔍 Resume Screener")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px; margin-bottom:24px;'>"
        "Upload a resume and select a job role to get an instant match score.</p>",
        unsafe_allow_html=True,
    )

    # Config row 
    col_role, col_spacer = st.columns([2, 3])
    with col_role:
        selected_role = st.selectbox("Job Role", AVAILABLE_ROLES, index=0)

    st.markdown("---")

    # Upload + paste tabs 
    tab_upload, tab_paste = st.tabs(["📁  Upload Resume (TXT)", "✏️  Paste Resume Text"])

    resume_text = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a plain text resume (.txt)",
            type=["txt"],
            help="PDF support requires pdfminer. For now, save your resume as .txt.",
        )
        if uploaded:
            resume_text = uploaded.read().decode("utf-8", errors="ignore")
            st.success(f"✓ File loaded — {len(resume_text):,} characters")

    with tab_paste:
        pasted = st.text_area(
            "Paste resume text here",
            height=220,
            placeholder="Copy and paste your resume content here...",
        )
        if pasted.strip():
            resume_text = pasted.strip()

    st.markdown("")

    run_btn = st.button("▶  Screen This Resume", use_container_width=True)

    # Results
    if run_btn:
        if not resume_text:
            st.warning("Please upload or paste a resume first.")
        else:
            with st.spinner("Analysing resume..."):
                result = score_single_resume(resume_text, selected_role)

            tier  = result["tier"]
            color = TIER_ST_COLORS.get(tier, "#6b7280")

            st.markdown("---")
            st.markdown("### Results")

            # Tier badge
            st.markdown(
                f"<div style='margin-bottom:20px;'>"
                f"<span style='background:{color}18; color:{color}; "
                f"border:1px solid {color}; padding:5px 16px; "
                f"border-radius:20px; font-size:13px; font-weight:600;'>"
                f"{tier}</span>"
                f"<span style='color:#64748b; font-size:13px; margin-left:12px;'>"
                f"vs {result['jd_title']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Score metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Score",      f"{result['final_score']}%")
            m2.metric("Text Similarity",  f"{result['cosine_score']}%")
            m3.metric("Skill Match",      f"{result['skill_match']}%")

            st.markdown("")

            # Skill breakdown
            sk1, sk2, sk3 = st.columns(3)

            with sk1:
                st.markdown("**✅ Matched Skills**")
                if result["matched_skills"]:
                    tags = " ".join([
                        f"<span class='skill-tag' style='background:#f0fdf4;"
                        f"color:#16a34a; border:1px solid #86efac;'>{s}</span>"
                        for s in result["matched_skills"]
                    ])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#9ca3af;'>None found</span>", unsafe_allow_html=True)

            with sk2:
                st.markdown("**❌ Missing Skills**")
                if result["missing_skills"]:
                    tags = " ".join([
                        f"<span class='skill-tag' style='background:#fef2f2;"
                        f"color:#dc2626; border:1px solid #fca5a5;'>{s}</span>"
                        for s in result["missing_skills"]
                    ])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#9ca3af;'>None missing</span>", unsafe_allow_html=True)

            with sk3:
                st.markdown("**➕ Extra Skills**")
                if result["extra_skills"]:
                    tags = " ".join([
                        f"<span class='skill-tag' style='background:#eff6ff;"
                        f"color:#3b82f6; border:1px solid #93c5fd;'>{s}</span>"
                        for s in result["extra_skills"][:8]
                    ])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#9ca3af;'>None</span>", unsafe_allow_html=True)

            st.markdown("")

            # JD required skills reference
            with st.expander("📋 View all required skills for this role"):
                tags = " ".join([
                    f"<span class='skill-tag' style='background:#f8fafc;"
                    f"color:#475569; border:1px solid #cbd5e1;'>{s}</span>"
                    for s in result["jd_skills"]
                ])
                st.markdown(tags if tags else "No skills extracted from JD.", unsafe_allow_html=True)

            # Score explanation
            with st.expander("ℹ️ How is this score calculated?"):
                st.markdown("""
                **Final Score** is a weighted average of two signals:

                - **Text Similarity (50%)** — TF-IDF cosine similarity between your resume
                  and the job description. Measures overall language and keyword overlap.

                - **Skill Match (50%)** — the percentage of required skills from the JD
                  that were found in your resume using spaCy's PhraseMatcher.

                ```
                Final Score = (Text Similarity × 0.5) + (Skill Match % × 0.5)
                ```

                | Tier | Score |
                |------|-------|
                | Strong Match | 40%+ |
                | Good Match | 25–39% |
                | Partial Match | 10–24% |
                | Weak Match | below 10% |
                """)


# PAGE 2 — ANALYTICS DASHBOARD

elif "Analytics" in page:

    st.markdown("## 📊 Analytics Dashboard")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px; margin-bottom:24px;'>"
        "Full analysis across all screened candidates for a role.</p>",
        unsafe_allow_html=True,
    )

    # Config 
    cfg1, cfg2, cfg3 = st.columns([2, 2, 1])
    with cfg1:
        dash_role = st.selectbox("Job Role", AVAILABLE_ROLES, index=0, key="dash_role")
    with cfg2:
        dash_cat  = st.selectbox("Resume Category", AVAILABLE_CATEGORIES, index=0, key="dash_cat")
    with cfg3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_dash = st.button("Run Analysis", use_container_width=True)

    st.markdown("---")

    if run_dash or "ranked_df" in st.session_state:

        if run_dash:
            with st.spinner("Running full pipeline..."):
                try:
                    ranked_df, jd, jd_skills = load_and_score_all(dash_cat, dash_role)
                    st.session_state["ranked_df"] = ranked_df
                    st.session_state["jd"]        = jd
                    st.session_state["jd_skills"] = jd_skills
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

        ranked_df = st.session_state["ranked_df"]
        jd        = st.session_state["jd"]
        jd_skills = st.session_state["jd_skills"]

        # Summary metrics
        total   = len(ranked_df)
        top_s   = ranked_df["Final_Score"].max()
        avg_s   = round(ranked_df["Final_Score"].mean(), 1)
        strong  = (ranked_df["Final_Score"] >= 40).sum()
        good    = ((ranked_df["Final_Score"] >= 25) & (ranked_df["Final_Score"] < 40)).sum()

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Candidates", total)
        s2.metric("Top Score",        f"{top_s}%")
        s3.metric("Average Score",    f"{avg_s}%")
        s4.metric("Strong Matches",   strong)
        s5.metric("Good Matches",     good)

        st.markdown("")

        # Charts row
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Score Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f8fafc")
            scores = ranked_df["Final_Score"]
            ax.hist(scores, bins=15, color="#3b82f6", alpha=0.8, edgecolor="white", linewidth=0.8)
            ax.axvline(scores.mean(), color="#ef4444", linestyle="--", linewidth=1.2, label=f"Avg: {avg_s}%")
            ax.set_xlabel("Final Score (%)", fontsize=10, color="#374151")
            ax.set_ylabel("Candidates",      fontsize=10, color="#374151")
            ax.tick_params(colors="#6b7280", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#e2e8f0")
            ax.legend(fontsize=9, framealpha=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with ch2:
            st.markdown("#### Candidates by Tier")
            tier_counts = ranked_df["Tier"].value_counts()
            tier_order  = ["Strong Match", "Good Match", "Partial Match", "Weak Match"]
            labels  = [t for t in tier_order if t in tier_counts.index]
            values  = [tier_counts[t] for t in labels]
            colors  = [TIER_ST_COLORS[t] for t in labels]

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            fig2.patch.set_facecolor("#ffffff")
            bars = ax2.barh(labels, values, color=colors, height=0.5)
            ax2.set_xlabel("Number of Candidates", fontsize=10, color="#374151")
            ax2.tick_params(colors="#374151", labelsize=9)
            ax2.set_facecolor("#f8fafc")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#e2e8f0")
            for bar, val in zip(bars, values):
                ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         str(val), va="center", fontsize=9, color="#374151")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")

        # Top candidates
        st.markdown("#### 🏆 Top 10 Candidates")

        top10 = ranked_df.head(10).copy()

        for _, row in top10.iterrows():
            tier  = row.get("Tier", "Weak Match")
            color = TIER_ST_COLORS.get(tier, "#6b7280")

            with st.container():
                c1, c2, c3, c4 = st.columns([1, 2, 3, 3])

                with c1:
                    st.markdown(
                        f"<div style='font-size:22px; font-weight:700; color:#1e293b;'>"
                        f"#{int(row['Rank'])}</div>",
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown(
                        f"<div style='font-size:13px; color:#64748b;'>Candidate</div>"
                        f"<div style='font-size:14px; font-weight:600; color:#1e293b; font-family:monospace;'>"
                        f"{row['Candidate_ID']}</div>"
                        f"<span style='background:{color}18; color:{color}; "
                        f"border:1px solid {color}; padding:2px 10px; "
                        f"border-radius:20px; font-size:11px; font-weight:600;'>"
                        f"{tier}</span>",
                        unsafe_allow_html=True,
                    )

                with c3:
                    st.markdown(
                        f"<div style='font-size:12px; color:#64748b; margin-bottom:4px;'>"
                        f"Final: <b style='color:#1e293b;'>{row['Final_Score']}%</b> &nbsp;|&nbsp; "
                        f"Text: <b>{row['Cosine_Score']}%</b> &nbsp;|&nbsp; "
                        f"Skills: <b>{row['Skill_Match']}%</b></div>",
                        unsafe_allow_html=True,
                    )
                    matched = row["Matched_Skills"]
                    if matched and matched != "None":
                        tags = " ".join([
                            f"<span class='skill-tag' style='background:#f0fdf4;"
                            f"color:#16a34a; border:1px solid #86efac; font-size:11px;'>{s.strip()}</span>"
                            for s in matched.split(",")[:5]
                        ])
                        st.markdown(tags, unsafe_allow_html=True)

                with c4:
                    missing = row["Missing_Skills"]
                    if missing and missing != "None":
                        st.markdown(
                            "<div style='font-size:11px; color:#64748b; margin-bottom:4px;'>Missing:</div>",
                            unsafe_allow_html=True,
                        )
                        tags = " ".join([
                            f"<span class='skill-tag' style='background:#fef2f2;"
                            f"color:#dc2626; border:1px solid #fca5a5; font-size:11px;'>{s.strip()}</span>"
                            for s in missing.split(",")[:5]
                        ])
                        st.markdown(tags, unsafe_allow_html=True)

                st.markdown(
                    "<hr style='margin:10px 0; border-color:#f1f5f9;'>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Full rankings table
        st.markdown("#### 📋 Full Rankings Table")

        display_cols = ["Rank", "Candidate_ID", "Tier", "Final_Score",
                        "Cosine_Score", "Skill_Match", "Matched_Skills", "Missing_Skills"]
        st.dataframe(
            ranked_df[display_cols].style.background_gradient(
                subset=["Final_Score"], cmap="Blues"
            ),
            use_container_width=True,
            height=400,
        )

        # Download button
        csv = ranked_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇  Download Full Results CSV",
            data      = csv,
            file_name = f"ranked_{jd['title'].replace(' ','_').lower()}.csv",
            mime      = "text/csv",
        )

        st.markdown("---")

        #JD skills reference 
        st.markdown("#### 📌 Required Skills for This Role")
        if jd_skills:
            tags = " ".join([
                f"<span class='skill-tag' style='background:#f8fafc;"
                f"color:#475569; border:1px solid #cbd5e1;'>{s}</span>"
                for s in sorted(jd_skills)
            ])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No skills extracted from the JD for this role.")

    else:
        st.info("Select a job role and resume category above, then click **Run Analysis** to start.")