import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT_ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
REPORTS = PROJECT_ROOT / "reports"

WEIGHTS = {
    "required_skill": 0.55,
    "nice_skill": 0.15,
    "tfidf": 0.20,
    "experience": 0.10,
}

SKILL_ALIASES = {
    "ms excel": "excel",
    "microsoft excel": "excel",
    "rest apis": "rest api",
    "rest api": "rest api",
    "scikit-learn": "scikit learn",
    "scikit learn": "scikit learn",
}


def normalize_skill(value):
    skill = str(value).strip().casefold()
    return SKILL_ALIASES.get(skill, skill)


def split_skills(value):
    if pd.isna(value):
        return set()
    return {
        normalize_skill(item)
        for item in re.split(r"[;,]", str(value))
        if item.strip()
    }


def clean_resume_text(value):
    """Remove the resume header and education section from free text."""
    if pd.isna(value):
        return ""

    lines = str(value).splitlines()

    # The first two lines contain the candidate name and contact/location.
    text = "\n".join(lines[2:])

    # Education text can contain college-tier information.
    text = re.sub(r"(?is)\n\s*EDUCATION\s*\n.*$", "", text)

    # Remove any remaining email addresses.
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text)

    return text


def experience_fit(experience, minimum):
    experience = pd.to_numeric(experience, errors="coerce")
    minimum = pd.to_numeric(minimum, errors="coerce")

    if pd.isna(experience):
        return 0.0
    if pd.isna(minimum) or minimum <= 0:
        return 1.0

    return float(min(experience / minimum, 1.0))


def load_data():
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")

    resumes = pd.read_excel(WORKBOOK, sheet_name="resumes")
    jobs = pd.read_excel(WORKBOOK, sheet_name="job_descriptions")
    labels = pd.read_excel(WORKBOOK, sheet_name="resume_job_labels")

    required_resume_columns = {
        "resume_id", "total_experience_years", "current_title",
        "highest_education", "skills", "certifications", "resume_text"
    }
    required_job_columns = {
        "job_id", "job_title", "min_experience_years",
        "required_skills", "nice_to_have_skills", "description_text"
    }
    required_label_columns = {
        "resume_id", "job_id", "human_suitability_score", "shortlist"
    }

    for name, frame, required in [
        ("resumes", resumes, required_resume_columns),
        ("jobs", jobs, required_job_columns),
        ("labels", labels, required_label_columns),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    if labels.duplicated(["resume_id", "job_id"]).any():
        raise ValueError("Duplicate resume-job labels found.")

    if not labels["resume_id"].isin(resumes["resume_id"]).all():
        raise ValueError("Some labels reference unknown resume IDs.")

    if not labels["job_id"].isin(jobs["job_id"]).all():
        raise ValueError("Some labels reference unknown job IDs.")

    return resumes, jobs, labels


def build_rankings(resumes, jobs):
    resume_texts = []
    for _, resume in resumes.iterrows():
        # Deliberately exclude candidate_name, email, city and college_tier.
        text = " ".join([
            str(resume.get("current_title", "")),
            str(resume.get("skills", "")),
            str(resume.get("certifications", "")),
            str(resume.get("highest_education", "")),
            clean_resume_text(resume.get("resume_text", "")),
        ])
        resume_texts.append(text)

    job_texts = []
    for _, job in jobs.iterrows():
        text = " ".join([
            str(job.get("job_title", "")),
            str(job.get("required_skills", "")),
            str(job.get("nice_to_have_skills", "")),
            str(job.get("description_text", "")),
        ])
        job_texts.append(text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )

    combined_texts = resume_texts + job_texts
    matrix = vectorizer.fit_transform(combined_texts)

    resume_matrix = matrix[:len(resumes)]
    job_matrix = matrix[len(resumes):]
    similarity = cosine_similarity(job_matrix, resume_matrix)

    rows = []

    for job_index, (_, job) in enumerate(jobs.iterrows()):
        required = split_skills(job["required_skills"])
        nice = split_skills(job["nice_to_have_skills"])

        for resume_index, (_, resume) in enumerate(resumes.iterrows()):
            candidate_skills = split_skills(resume["skills"])

            required_coverage = (
                len(required & candidate_skills) / len(required)
                if required else 0.0
            )
            nice_coverage = (
                len(nice & candidate_skills) / len(nice)
                if nice else 0.0
            )

            tfidf_score = float(similarity[job_index, resume_index])
            exp_score = experience_fit(
                resume["total_experience_years"],
                job["min_experience_years"],
            )

            final_score = (
                WEIGHTS["required_skill"] * required_coverage
                + WEIGHTS["nice_skill"] * nice_coverage
                + WEIGHTS["tfidf"] * tfidf_score
                + WEIGHTS["experience"] * exp_score
            )

            rows.append({
                "resume_id": resume["resume_id"],
                "job_id": job["job_id"],
                "job_title": job["job_title"],
                "total_experience_years": resume["total_experience_years"],
                "required_skill_coverage": required_coverage,
                "nice_skill_coverage": nice_coverage,
                "tfidf_similarity": tfidf_score,
                "experience_fit": exp_score,
                "ranking_score": final_score,
            })

    rankings = pd.DataFrame(rows)
    rankings = rankings.sort_values(
        ["job_id", "ranking_score", "resume_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    rankings["overall_rank"] = (
        rankings.groupby("job_id").cumcount() + 1
    )

    return rankings


def evaluate_rankings(rankings, labels):
    labels = labels.rename(columns={
        "required_skill_coverage": "human_required_skill_coverage",
        "nice_skill_coverage": "human_nice_skill_coverage",
    })

    labelled = rankings.merge(
        labels,
        on=["resume_id", "job_id"],
        how="inner",
        validate="one_to_one",
    )

    labelled["shortlist_binary"] = (
        labelled["shortlist"].astype(str).str.upper() == "Y"
    ).astype(int)

    # Rank only the labelled resumes when calculating labelled top-5 metrics.
    labelled = labelled.sort_values(
        ["job_id", "ranking_score", "resume_id"],
        ascending=[True, False, True],
    ).copy()

    labelled["labelled_rank"] = (
        labelled.groupby("job_id").cumcount() + 1
    )

    if labelled["shortlist_binary"].nunique() == 2:
        auc = roc_auc_score(
            labelled["shortlist_binary"], labelled["ranking_score"]
        )
        average_precision = average_precision_score(
            labelled["shortlist_binary"], labelled["ranking_score"]
        )
    else:
        auc = np.nan
        average_precision = np.nan

    job_metrics = []

    for job_id, group in labelled.groupby("job_id"):
        top5 = group.nsmallest(5, "labelled_rank")
        hits = int(top5["shortlist_binary"].sum())
        total_shortlisted = int(group["shortlist_binary"].sum())

        ndcg = ndcg_score(
            [group["human_suitability_score"].to_numpy()],
            [group["ranking_score"].to_numpy()],
            k=5,
        )

        job_metrics.append({
            "job_id": job_id,
            "labelled_candidates": len(group),
            "shortlisted_candidates": total_shortlisted,
            "shortlist_hits_at_5": hits,
            "precision_at_5": hits / 5,
            "recall_at_5": (
                hits / total_shortlisted if total_shortlisted else 0.0
            ),
            "ndcg_at_5": ndcg,
        })

    metrics_by_job = pd.DataFrame(job_metrics)

    summary = pd.DataFrame([{
        "labelled_pairs": len(labelled),
        "shortlisted_pairs": int(labelled["shortlist_binary"].sum()),
        "not_shortlisted_pairs": int(
            (labelled["shortlist_binary"] == 0).sum()
        ),
        "roc_auc": auc,
        "average_precision": average_precision,
        "mean_precision_at_5": metrics_by_job["precision_at_5"].mean(),
        "mean_recall_at_5": metrics_by_job["recall_at_5"].mean(),
        "mean_ndcg_at_5": metrics_by_job["ndcg_at_5"].mean(),
        "jobs_with_shortlist_in_top_5": int(
            (metrics_by_job["shortlist_hits_at_5"] > 0).sum()
        ),
        "jobs_evaluated": len(metrics_by_job),
    }])

    return labelled, summary, metrics_by_job


def create_misranking_evidence(labelled):
    # First select shortlisted candidates that the model placed below top 5.
    missed = labelled[
        (labelled["shortlist_binary"] == 1)
        & (labelled["labelled_rank"] > 5)
    ].copy()

    missed = missed.sort_values(
        ["labelled_rank", "ranking_score"],
        ascending=[False, True],
    )

    # If fewer than five missed shortlisted candidates exist, add
    # non-shortlisted candidates that appeared in the labelled top five.
    if len(missed) < 5:
        additional = labelled[
            (labelled["shortlist_binary"] == 0)
            & (labelled["labelled_rank"] <= 5)
        ].copy()
        additional = additional.sort_values(
            "ranking_score", ascending=False
        )
        missed = pd.concat(
            [missed, additional.head(5 - len(missed))],
            ignore_index=True,
        )

    missed = missed.head(5).copy()

    missed["review_observation"] = missed.apply(
        lambda row: (
            f"Required skills matched: "
            f"{row['required_skill_coverage']:.0%}; "
            f"nice-to-have skills matched: "
            f"{row['nice_skill_coverage']:.0%}; "
            f"TF-IDF similarity: {row['tfidf_similarity']:.3f}; "
            f"experience fit: {row['experience_fit']:.0%}. "
            "Review these components against the job requirements."
        ),
        axis=1,
    )

    columns = [
        "resume_id", "job_id", "job_title", "shortlist",
        "human_suitability_score", "labelled_rank", "ranking_score",
        "required_skill_coverage", "nice_skill_coverage",
        "tfidf_similarity", "experience_fit", "review_observation",
    ]
    return missed[columns]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading Task 11 datasets...")
    resumes, jobs, labels = load_data()
    print(f"Loaded {len(resumes)} resumes, {len(jobs)} jobs, "
          f"and {len(labels)} human labels.")

    print("Building resume rankings...")
    rankings = build_rankings(resumes, jobs)

    print("Evaluating against human labels...")
    labelled, summary, metrics_by_job = evaluate_rankings(
        rankings, labels
    )

    misranked = create_misranking_evidence(labelled)

    rankings.to_csv(REPORTS / "task11_resume_rankings.csv", index=False)
    labelled.to_csv(
        REPORTS / "task11_labelled_evaluation.csv", index=False
    )
    summary.to_csv(
        REPORTS / "task11_evaluation_summary.csv", index=False
    )
    metrics_by_job.to_csv(
        REPORTS / "task11_top5_metrics_by_job.csv", index=False
    )
    misranked.to_csv(
        REPORTS / "task11_misranked_examples.csv", index=False
    )

    report = [
        "# Task 11 – Resume Ranking",
        "",
        "## Method",
        "",
        "All resumes were ranked against all job descriptions.",
        "The score combines required-skill coverage (55%), "
        "nice-to-have skill coverage (15%), TF-IDF similarity (20%), "
        "and minimum-experience suitability (10%).",
        "",
        "Candidate names, email addresses, locations, and college tier "
        "were excluded from the scoring inputs.",
        "",
        "These are initial heuristic weights, not tuned weights.",
        "",
        "## Overall evaluation",
        "",
        summary.to_string(index=False),
        "",
        "## Top-5 metrics by job",
        "",
        metrics_by_job.to_string(index=False),
        "",
        "## Misranked examples for manual review",
        "",
        misranked.to_string(index=False),
        "",
        "## Interpretation",
        "",
        "Top-5 metrics are calculated within the labelled candidates "
        "for each job. Unlabelled resume-job combinations are not "
        "treated as rejected candidates.",
        "",
        "The ranking score is a decision-support signal, not an "
        "automated hiring decision. Human review is required.",
    ]

    (REPORTS / "task11_resume_ranking_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n========== TASK 11 OVERALL METRICS ==========")
    print(summary.to_string(index=False))

    print("\n========== TOP-5 METRICS BY JOB ==========")
    print(metrics_by_job.to_string(index=False))

    print("\n========== FIVE REVIEW EXAMPLES ==========")
    print(misranked.to_string(index=False))

    print("\nSaved Task 11 evidence:")
    for filename in [
        "task11_resume_rankings.csv",
        "task11_labelled_evaluation.csv",
        "task11_evaluation_summary.csv",
        "task11_top5_metrics_by_job.csv",
        "task11_misranked_examples.csv",
        "task11_resume_ranking_report.md",
    ]:
        print(" -", REPORTS / filename)


if __name__ == "__main__":
    main()
