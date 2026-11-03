import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.task11_resume_ranking import (
    PROJECT_ROOT,
    REPORTS,
    WEIGHTS,
    load_data,
    build_rankings,
    evaluate_rankings,
    split_skills,
    experience_fit,
)

from src.cortexhr_helpers import group_gap


UNLOCK_FILE = (
    PROJECT_ROOT / "data" / "unlocked"
    / "unlock_CX-12_resume_protected_attrs.csv"
)


def build_unmasked_diagnostic(resumes, jobs):
    """Diagnostic only: deliberately includes protected/proxy information."""

    resume_texts = []

    for _, resume in resumes.iterrows():
        text = " ".join([
            str(resume.get("candidate_name", "")),
            str(resume.get("city", "")),
            str(resume.get("college_tier", "")),
            str(resume.get("current_title", "")),
            str(resume.get("skills", "")),
            str(resume.get("certifications", "")),
            str(resume.get("highest_education", "")),
            str(resume.get("resume_text", "")),
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

    matrix = vectorizer.fit_transform(resume_texts + job_texts)
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


def group_diagnostics(rankings, labels, protected, method):
    labelled = rankings.merge(
        labels,
        on=["resume_id", "job_id"],
        how="inner",
        validate="one_to_one",
    )

    labelled["human_selected"] = (
        labelled["shortlist"].astype(str).str.upper() == "Y"
    ).astype(int)

    labelled = labelled.sort_values(
        ["job_id", "ranking_score", "resume_id"],
        ascending=[True, False, True],
    ).copy()

    labelled["labelled_rank"] = (
        labelled.groupby("job_id").cumcount() + 1
    )
    labelled["model_selected"] = (
        labelled["labelled_rank"] <= 5
    ).astype(int)

    labelled = labelled.merge(
        protected,
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    results = []

    for group_column in ["inferred_gender", "college_tier"]:
        data = labelled.dropna(subset=[group_column]).copy()

        for outcome in ["human_selected", "model_selected"]:
            gap_result = group_gap(
                data, group_column, outcome
            )

            rates = (
                data.groupby(group_column)[outcome]
                .agg(["mean", "count"])
                .reset_index()
            )

            for _, row in rates.iterrows():
                results.append({
                    "method": method,
                    "group_type": group_column,
                    "group": row[group_column],
                    "outcome": outcome,
                    "selection_rate": row["mean"],
                    "count": int(row["count"]),
                    "gap": gap_result["gap"],
                })

    return pd.DataFrame(results)


def main():
    REPORTS.mkdir(exist_ok=True)

    if not UNLOCK_FILE.exists():
        raise FileNotFoundError(
            f"Protected attributes file not found: {UNLOCK_FILE}"
        )

    resumes, jobs, labels = load_data()
    protected = pd.read_csv(UNLOCK_FILE)

    print("Building masked rankings...")
    masked = build_rankings(resumes, jobs)

    print("Building unmasked diagnostic rankings...")
    unmasked = build_unmasked_diagnostic(resumes, jobs)

    print("Evaluating both versions...")
    _, masked_summary, _ = evaluate_rankings(masked, labels)
    _, unmasked_summary, _ = evaluate_rankings(unmasked, labels)

    masked_summary.insert(0, "method", "masked")
    unmasked_summary.insert(0, "method", "unmasked_diagnostic")

    comparison = pd.concat(
        [masked_summary, unmasked_summary],
        ignore_index=True,
    )

    comparison.to_csv(
        REPORTS / "task12_masking_comparison.csv",
        index=False,
    )

    print("Calculating protected-group selection rates...")
    masked_groups = group_diagnostics(
        masked, labels, protected, "masked"
    )
    unmasked_groups = group_diagnostics(
        unmasked, labels, protected, "unmasked_diagnostic"
    )

    group_results = pd.concat(
        [masked_groups, unmasked_groups],
        ignore_index=True,
    )

    group_results.to_csv(
        REPORTS / "task12_group_rates_masked_unmasked.csv",
        index=False,
    )

    report = [
        "# Task 12 - Masking and Fairness Mitigation Comparison",
        "",
        "## Purpose",
        "",
        "Compare the existing masked ranking model with an "
        "unmasked diagnostic-only version.",
        "",
        "The unmasked version deliberately includes candidate name, "
        "city, college tier and raw resume text. It is not intended "
        "for production hiring or candidate selection.",
        "",
        "## Ranking quality comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Protected-group selection rates",
        "",
        group_results.to_markdown(index=False),
        "",
        "## Interpretation and limitations",
        "",
        "- The masked version excludes direct identifiers and "
        "college tier from its resume text.",
        "- The unmasked version is a diagnostic experiment, not "
        "a recommended hiring model.",
        "- Differences in group selection rates do not by themselves "
        "prove discrimination or fairness.",
        "- The dataset contains a limited number of labelled "
        "candidates and supplied counterfactual pairs.",
        "- Task 11 evaluation labels closely reflect skill coverage, "
        "so ranking metrics may be optimistic and are not independent "
        "validation of hiring quality.",
        "- Human review remains necessary before any hiring decision.",
        "",
    ]

    (REPORTS / "task12_masking_comparison_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nComparison completed.")
    print("\nRanking quality:")
    print(comparison.to_string(index=False))

    print("\nSaved:")
    print("reports/task12_masking_comparison.csv")
    print("reports/task12_group_rates_masked_unmasked.csv")
    print("reports/task12_masking_comparison_report.md")


if __name__ == "__main__":
    main()
