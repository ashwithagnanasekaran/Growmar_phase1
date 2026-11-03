# Task 12 - Masking and Fairness Mitigation Comparison

## Purpose

Compare the existing masked ranking model with an unmasked diagnostic-only version.

The unmasked version deliberately includes candidate name, city, college tier and raw resume text. It is not intended for production hiring or candidate selection.

## Ranking quality comparison

| method              |   labelled_pairs |   shortlisted_pairs |   not_shortlisted_pairs |   roc_auc |   average_precision |   mean_precision_at_5 |   mean_recall_at_5 |   mean_ndcg_at_5 |   jobs_with_shortlist_in_top_5 |   jobs_evaluated |
|:--------------------|-----------------:|--------------------:|------------------------:|----------:|--------------------:|----------------------:|-------------------:|-----------------:|-------------------------------:|-----------------:|
| masked              |              555 |                  95 |                     460 |  0.998261 |            0.992685 |                  0.96 |           0.582459 |         0.986989 |                             10 |               10 |
| unmasked_diagnostic |              555 |                  95 |                     460 |  0.998261 |            0.992677 |                  0.96 |           0.582459 |         0.986989 |                             10 |               10 |

## Protected-group selection rates

| method              | group_type      | group   | outcome        |   selection_rate |   count |        gap |
|:--------------------|:----------------|:--------|:---------------|-----------------:|--------:|-----------:|
| masked              | inferred_gender | Female  | human_selected |        0.184685  |     222 | 0.0225225  |
| masked              | inferred_gender | Male    | human_selected |        0.162162  |     333 | 0.0225225  |
| masked              | inferred_gender | Female  | model_selected |        0.0900901 |     222 | 0          |
| masked              | inferred_gender | Male    | model_selected |        0.0900901 |     333 | 0          |
| masked              | college_tier    | Tier 1  | human_selected |        0.197279  |     147 | 0.051739   |
| masked              | college_tier    | Tier 2  | human_selected |        0.14554   |     213 | 0.051739   |
| masked              | college_tier    | Tier 3  | human_selected |        0.179487  |     195 | 0.051739   |
| masked              | college_tier    | Tier 1  | model_selected |        0.102041  |     147 | 0.0222286  |
| masked              | college_tier    | Tier 2  | model_selected |        0.0798122 |     213 | 0.0222286  |
| masked              | college_tier    | Tier 3  | model_selected |        0.0923077 |     195 | 0.0222286  |
| unmasked_diagnostic | inferred_gender | Female  | human_selected |        0.184685  |     222 | 0.0225225  |
| unmasked_diagnostic | inferred_gender | Male    | human_selected |        0.162162  |     333 | 0.0225225  |
| unmasked_diagnostic | inferred_gender | Female  | model_selected |        0.0945946 |     222 | 0.00750751 |
| unmasked_diagnostic | inferred_gender | Male    | model_selected |        0.0870871 |     333 | 0.00750751 |
| unmasked_diagnostic | college_tier    | Tier 1  | human_selected |        0.197279  |     147 | 0.051739   |
| unmasked_diagnostic | college_tier    | Tier 2  | human_selected |        0.14554   |     213 | 0.051739   |
| unmasked_diagnostic | college_tier    | Tier 3  | human_selected |        0.179487  |     195 | 0.051739   |
| unmasked_diagnostic | college_tier    | Tier 1  | model_selected |        0.102041  |     147 | 0.0222286  |
| unmasked_diagnostic | college_tier    | Tier 2  | model_selected |        0.0798122 |     213 | 0.0222286  |
| unmasked_diagnostic | college_tier    | Tier 3  | model_selected |        0.0923077 |     195 | 0.0222286  |

## Interpretation and limitations

- The masked version excludes direct identifiers and college tier from its resume text.
- The unmasked version is a diagnostic experiment, not a recommended hiring model.
- Differences in group selection rates do not by themselves prove discrimination or fairness.
- The dataset contains a limited number of labelled candidates and supplied counterfactual pairs.
- Task 11 evaluation labels closely reflect skill coverage, so ranking metrics may be optimistic and are not independent validation of hiring quality.
- Human review remains necessary before any hiring decision.
