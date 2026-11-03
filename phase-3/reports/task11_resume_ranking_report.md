# Task 11 – Resume Ranking

## Method

All resumes were ranked against all job descriptions.
The score combines required-skill coverage (55%), nice-to-have skill coverage (15%), TF-IDF similarity (20%), and minimum-experience suitability (10%).

Candidate names, email addresses, locations, and college tier were excluded from the scoring inputs.

These are initial heuristic weights, not tuned weights.

## Overall evaluation

 labelled_pairs  shortlisted_pairs  not_shortlisted_pairs  roc_auc  average_precision  mean_precision_at_5  mean_recall_at_5  mean_ndcg_at_5  jobs_with_shortlist_in_top_5  jobs_evaluated
            555                 95                    460 0.998261           0.992685                 0.96          0.582459        0.986989                            10              10

## Top-5 metrics by job

job_id  labelled_candidates  shortlisted_candidates  shortlist_hits_at_5  precision_at_5  recall_at_5  ndcg_at_5
   J01                   64                       7                    5             1.0     0.714286   1.000000
   J02                   47                       3                    3             0.6     1.000000   0.989692
   J03                   60                      12                    5             1.0     0.416667   1.000000
   J04                   62                      16                    5             1.0     0.312500   0.926049
   J05                   57                       9                    5             1.0     0.555556   1.000000
   J06                   49                       6                    5             1.0     0.833333   1.000000
   J07                   49                       8                    5             1.0     0.625000   1.000000
   J08                   55                       9                    5             1.0     0.555556   1.000000
   J09                   57                      14                    5             1.0     0.357143   0.957203
   J10                   55                      11                    5             1.0     0.454545   0.996947

## Misranked examples for manual review

resume_id job_id         job_title shortlist  human_suitability_score  labelled_rank  ranking_score  required_skill_coverage  nice_skill_coverage  tfidf_similarity  experience_fit                                                                                                                                                    review_observation
 RES-0181    J04       ML Engineer         Y                        4             18       0.530566                      0.6                  0.4          0.202832             1.0 Required skills matched: 60%; nice-to-have skills matched: 40%; TF-IDF similarity: 0.203; experience fit: 100%. Review these components against the job requirements.
 RES-0055    J09 Account Executive         Y                        4             17       0.461193                      0.6                  0.0          0.155966             1.0  Required skills matched: 60%; nice-to-have skills matched: 0%; TF-IDF similarity: 0.156; experience fit: 100%. Review these components against the job requirements.
 RES-0034    J04       ML Engineer         Y                        4             17       0.530566                      0.6                  0.4          0.202832             1.0 Required skills matched: 60%; nice-to-have skills matched: 40%; TF-IDF similarity: 0.203; experience fit: 100%. Review these components against the job requirements.
 RES-0125    J09 Account Executive         Y                        4             15       0.538449                      0.6                  0.5          0.217245             0.9  Required skills matched: 60%; nice-to-have skills matched: 50%; TF-IDF similarity: 0.217; experience fit: 90%. Review these components against the job requirements.
 RES-0180    J04       ML Engineer         Y                        4             14       0.595874                      0.8                  0.0          0.279368             1.0  Required skills matched: 80%; nice-to-have skills matched: 0%; TF-IDF similarity: 0.279; experience fit: 100%. Review these components against the job requirements.

## Interpretation

Top-5 metrics are calculated within the labelled candidates for each job. Unlabelled resume-job combinations are not treated as rejected candidates.

The ranking score is a decision-support signal, not an automated hiring decision. Human review is required.
## Manual review of five misranked candidates

The following cases were inspected against the original resume and job description.

| Resume | Job | Manual observation |
|---|---|---|
| RES-0181 | J04 | Three of five required skills are listed explicitly. Model deployment appears in experience but not in the skills field. |
| RES-0055 | J09 | Unusual resume formatting may reduce text similarity. Work history shows lead generation, account management and deal closure. |
| RES-0034 | J04 | Substantively identical resume content to RES-0181. Different college tier does not change the ranking score. |
| RES-0125 | J09 | Has B2B Sales, Negotiation and Lead Generation. CRM and Presentation Skills are absent from the skills field. Experience is below the stated minimum. |
| RES-0180 | J04 | Four of five required skills are listed. Relevant deployment experience exists, but TF-IDF similarity is relatively low. Resume text is uppercase. |

### Lessons from manual review

1. Skill extraction should consider both the dedicated skills field and experience descriptions.
2. Resume formatting can affect text similarity and should be handled carefully.
3. Experience below a job's minimum should be treated as a suitability consideration, not an automatic rejection.
4. College tier and candidate identity should not influence ranking.
5. Human review is necessary when the resume contains evidence that structured skill extraction misses.

These observations are based on five selected cases, not a complete audit of all candidates.
