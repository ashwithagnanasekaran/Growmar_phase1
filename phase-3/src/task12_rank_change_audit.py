import pandas as pd

from src.task11_resume_ranking import load_data, build_rankings
from src.task12_masking_comparison import build_unmasked_diagnostic
from src.cortexhr_helpers import group_gap

from src.task11_resume_ranking import PROJECT_ROOT, REPORTS

UNLOCK_FILE = (
    PROJECT_ROOT / "data" / "unlocked"
    / "unlock_CX-12_resume_protected_attrs.csv"
)


def main():
    resumes, jobs, labels = load_data()
    protected = pd.read_csv(UNLOCK_FILE)

    masked = build_rankings(resumes, jobs)
    unmasked = build_unmasked_diagnostic(resumes, jobs)

    # Compare scores and ranks for every resume-job combination.
    comparison = masked[
        ["resume_id", "job_id", "ranking_score", "overall_rank"]
    ].merge(
        unmasked[
            ["resume_id", "job_id", "ranking_score", "overall_rank"]
        ],
        on=["resume_id", "job_id"],
        suffixes=("_masked", "_unmasked"),
        validate="one_to_one",
    )

    comparison["absolute_rank_change"] = (
        comparison["overall_rank_masked"]
        - comparison["overall_rank_unmasked"]
    ).abs()

    comparison["absolute_score_change"] = (
        comparison["ranking_score_masked"]
        - comparison["ranking_score_unmasked"]
    ).abs()

    comparison.to_csv(
        REPORTS / "task12_rank_changes_all.csv",
        index=False,
    )

    # Compare top-five selection using labelled candidates only.
    labelled = labels[["resume_id", "job_id", "shortlist"]].copy()

    for method, rankings in [
        ("masked", masked),
        ("unmasked_diagnostic", unmasked),
    ]:
        selected = rankings.merge(
            labelled,
            on=["resume_id", "job_id"],
            how="inner",
            validate="one_to_one",
        )

        selected = selected.sort_values(
            ["job_id", "ranking_score", "resume_id"],
            ascending=[True, False, True],
        ).copy()

        selected["labelled_rank"] = (
            selected.groupby("job_id").cumcount() + 1
        )
        selected[f"{method}_selected"] = (
            selected["labelled_rank"] <= 5
        ).astype(int)

        if method == "masked":
            result = selected[
                ["resume_id", "job_id", "masked_selected"]
            ]
        else:
            result = selected[
                ["resume_id", "job_id", "unmasked_diagnostic_selected"]
            ]
            result = result.merge(
                masked_flags,
                on=["resume_id", "job_id"],
                validate="one_to_one",
            )

        if method == "masked":
            masked_flags = result.copy()
        else:
            final_selection = result

    final_selection = final_selection.merge(
        protected,
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    final_selection.to_csv(
        REPORTS / "task12_top5_selection_changes.csv",
        index=False,
    )

    # Job-level group rates and gaps.
    gap_rows = []

    for job_id, group in final_selection.groupby("job_id"):
        for group_column in ["inferred_gender", "college_tier"]:
            for outcome in [
                "masked_selected",
                "unmasked_diagnostic_selected",
            ]:
                gap = group_gap(group, group_column, outcome)

                rates = group.groupby(group_column)[outcome].agg(
                    ["mean", "count"]
                ).reset_index()

                for _, row in rates.iterrows():
                    gap_rows.append({
                        "job_id": job_id,
                        "group_type": group_column,
                        "group": row[group_column],
                        "method": outcome,
                        "selection_rate": row["mean"],
                        "count": int(row["count"]),
                        "gap": gap["gap"],
                    })

    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(
        REPORTS / "task12_job_level_group_gaps.csv",
        index=False,
    )

    print("Ranking changes:")
    print("Total resume-job comparisons:", len(comparison))
    print(
        "Comparisons with rank changes:",
        int((comparison["absolute_rank_change"] > 0).sum()),
    )
    print(
        "Mean absolute rank change:",
        round(comparison["absolute_rank_change"].mean(), 4),
    )
    print(
        "Maximum absolute rank change:",
        int(comparison["absolute_rank_change"].max()),
    )
    print(
        "Mean absolute score change:",
        round(comparison["absolute_score_change"].mean(), 6),
    )

    print("\nTop-five selection changes:")
    print(
        "Candidates whose top-five status changed:",
        int((
            final_selection["masked_selected"]
            != final_selection["unmasked_diagnostic_selected"]
        ).sum()),
    )

    print("\nJob-level group gaps saved.")

    print("\nCreated:")
    print("reports/task12_rank_changes_all.csv")
    print("reports/task12_top5_selection_changes.csv")
    print("reports/task12_job_level_group_gaps.csv")


if __name__ == "__main__":
    main()
