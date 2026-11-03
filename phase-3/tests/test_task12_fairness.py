import pandas as pd
import pytest

from src.cortexhr_helpers import group_gap
from src.task11_resume_ranking import build_rankings


def test_group_gap_calculates_correct_difference():
    data = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "selected": [1, 0, 1, 1],
    })

    result = group_gap(data, "group", "selected")

    assert result["gap"] == 0.5
    assert result["highest_group"] == "B"
    assert result["lowest_group"] == "A"


def test_group_gap_requires_two_groups():
    data = pd.DataFrame({
        "group": ["A", "A"],
        "selected": [1, 0],
    })

    with pytest.raises(ValueError):
        group_gap(data, "group", "selected")


def test_masked_rankings_ignore_protected_fields():
    resumes = pd.DataFrame([
        {
            "resume_id": "R001",
            "candidate_name": "Candidate One",
            "email": "one@example.com",
            "city": "Chennai",
            "college_tier": "Tier 1",
            "current_title": "Python Developer",
            "skills": "Python; SQL",
            "certifications": "Python Certificate",
            "highest_education": "B.Tech",
            "resume_text": "Python developer with SQL experience",
            "total_experience_years": 3,
        },
        {
            "resume_id": "R002",
            "candidate_name": "Candidate Two",
            "email": "two@example.com",
            "city": "Coimbatore",
            "college_tier": "Tier 3",
            "current_title": "Data Analyst",
            "skills": "Python; Excel",
            "certifications": "Data Analytics Certificate",
            "highest_education": "B.Sc",
            "resume_text": "Data analyst with Python experience",
            "total_experience_years": 2,
        },
    ])

    jobs = pd.DataFrame([
        {
            "job_id": "J001",
            "job_title": "Python Developer",
            "min_experience_years": 2,
            "required_skills": "Python; SQL",
            "nice_to_have_skills": "Excel",
            "description_text": "Develop applications using Python and SQL",
        }
    ])

    original = build_rankings(resumes, jobs)

    changed = resumes.copy()
    changed["candidate_name"] = ["Different Name A", "Different Name B"]
    changed["email"] = ["new_a@example.com", "new_b@example.com"]
    changed["city"] = ["Delhi", "Mumbai"]
    changed["college_tier"] = ["Tier 3", "Tier 1"]

    modified = build_rankings(changed, jobs)

    original_scores = original.set_index("resume_id")["ranking_score"]
    modified_scores = modified.set_index("resume_id")["ranking_score"]

    pd.testing.assert_series_equal(
        original_scores.sort_index(),
        modified_scores.sort_index(),
    )

    assert "candidate_name" not in original.columns
    assert "email" not in original.columns
    assert "city" not in original.columns
    assert "college_tier" not in original.columns
