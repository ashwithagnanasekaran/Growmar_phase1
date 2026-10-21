
"""Safely connect monthly features to both existing build scripts."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

IMPORT_OLD = "from src.features.pipeline import FeaturePipeline"
IMPORT_NEW = (
    "from src.features.pipeline import FeaturePipeline\n"
    "from src.features.monthly_aggregation import "
    "aggregate_monthly_features"
)

BASELINE_OLD = '''    df = pd.read_excel(
        DATA_PATH,
        sheet_name="attrition_dataset",
    )
'''

BASELINE_NEW = '''    df = pd.read_excel(
        DATA_PATH,
        sheet_name="attrition_dataset",
    )

    attendance_df = pd.read_excel(
        DATA_PATH,
        sheet_name="attendance_monthly_features",
    )
    payroll_df = pd.read_excel(
        DATA_PATH,
        sheet_name="payroll_monthly_features",
    )

    df = aggregate_monthly_features(
        df, attendance_df, payroll_df
    )
    print(f"Rows after monthly aggregation: {len(df)}")
'''

BUILDER_OLD = (
    '    df = pd.read_excel(DATA_PATH, sheet_name="attrition_dataset")\n'
)

BUILDER_NEW = '''    df = pd.read_excel(
        DATA_PATH, sheet_name="attrition_dataset"
    )
    attendance_df = pd.read_excel(
        DATA_PATH, sheet_name="attendance_monthly_features"
    )
    payroll_df = pd.read_excel(
        DATA_PATH, sheet_name="payroll_monthly_features"
    )
    df = aggregate_monthly_features(
        df, attendance_df, payroll_df
    )
    print(f"Rows after monthly aggregation: {len(df)}")
'''


def update_file(path, old_block, new_block):
    text = path.read_text(encoding="utf-8")

    if new_block in text:
        print(f"Already updated: {path.relative_to(ROOT)}")
        return

    if text.count(old_block) != 1:
        raise RuntimeError(
            f"Expected code block not found exactly once in {path}. "
            "No changes made to this file."
        )

    backup = path.with_suffix(path.suffix + ".backup")
    shutil.copy2(path, backup)

    updated = text.replace(old_block, new_block, 1)

    if path.name in {"baseline_models.py", "build.py"}:
        if IMPORT_OLD not in updated:
            raise RuntimeError(
                f"FeaturePipeline import not found in {path}"
            )
        updated = updated.replace(IMPORT_OLD, IMPORT_NEW, 1)

    path.write_text(updated, encoding="utf-8")
    print(f"Updated: {path.relative_to(ROOT)}")
    print(f"Backup:  {backup.name}")


def main():
    update_file(
        ROOT / "src" / "baseline_models.py",
        BASELINE_OLD,
        BASELINE_NEW,
    )
    update_file(
        ROOT / "cortexhr_features" / "build.py",
        BUILDER_OLD,
        BUILDER_NEW,
    )
    print("Update completed.")


if __name__ == "__main__":
    main()
