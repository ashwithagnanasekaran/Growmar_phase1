from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "AIML_Ashwitha_Phase1_Report.pdf"
REPO_URL = "https://github.com/ashwithagnanasekaran/Growmar_phase1"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=7
))
styles.add(ParagraphStyle(
    name="SubTitle", parent=styles["Normal"],
    fontSize=9, leading=13, alignment=TA_CENTER, spaceAfter=10
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"],
    fontSize=12, leading=15, spaceBefore=5, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="SmallBody", parent=styles["BodyText"],
    fontSize=8.3, leading=11, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="Tiny", parent=styles["BodyText"],
    fontSize=7.2, leading=9
))

def read_csv(name):
    path = REPORTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required evidence file missing: {path}")
    return pd.read_csv(path)

def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)

def make_table(rows, widths=None, font_size=7.5):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table

def add_image(story, filename, width=150*mm, height=76*mm):
    path = REPORTS / filename
    if path.exists():
        img = Image(str(path))
        ratio = min(width / img.imageWidth, height / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        story.append(img)
    else:
        story.append(Paragraph(
            f"Chart not found: {filename}", styles["SmallBody"]
        ))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18*mm, 10*mm, "AIML Phase 1 | Evidence-based project report")
    canvas.drawRightString(192*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()

def main():
    baseline = read_csv("baseline_metrics.csv")
    benchmark = read_csv("baseline_benchmark_comparison.csv")
    attr_metrics = read_csv("attrition_evaluation_metrics.csv")
    bands = read_csv("attrition_risk_band_counts.csv")
    anomaly = read_csv("attendance_validation_metrics.csv")
    anomaly_types = read_csv("attendance_validation_by_type.csv")

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        rightMargin=17*mm, leftMargin=17*mm,
        topMargin=14*mm, bottomMargin=17*mm,
        title="AIML Ashwitha Phase 1 Report",
        author="AIML Phase 1 participant",
    )
    story = []

    # PAGE 1: overview, data audit, feature engineering, baseline models
    story.append(Paragraph("AIML Phase 1 — Project Report", styles["ReportTitle"]))
    story.append(Paragraph(
        "Name: Ashwitha | Track: AIML | Phase: 1<br/>"
        f"Repository: <link href='{REPO_URL}' color='#2563EB'>{REPO_URL}</link>"
        " | Git tag: phase-1",
        styles["SubTitle"]
    ))
    story.append(Paragraph("Executive summary", styles["Section"]))
    story.append(Paragraph(
        "This phase covered data quality auditing, feature engineering, "
        "baseline attrition modelling, risk evaluation, and unsupervised "
        "attendance-anomaly validation. The work includes reproducible scripts, "
        "automated tests, CSV evidence, charts, and written interpretations. "
        "Results are experimental and require further independent validation "
        "before operational use.",
        styles["SmallBody"]
    ))

    story.append(Paragraph("Task completion and evidence", styles["Section"]))
    tasks = [
        ["Task", "Work completed", "Evidence"],
        ["1. Data audit",
         "Inspected data quality, duplicate employee/date pairs, invalid values, "
         "target distribution, and planted issues.",
         "data-audit.md; data-audit-overview.png"],
        ["2. Feature engineering",
         "Built preprocessing and monthly aggregation components; added tests "
         "for model-ready features and leakage exclusions.",
         "src/features; TASK2_EVIDENCE.md; tests"],
        ["3. Baseline modelling",
         "Compared Dummy, balanced Logistic Regression, and depth-4 Decision Tree "
         "using stratified five-fold cross-validation.",
         "baseline-report.md; baseline_metrics.csv; benchmark comparison"],
        ["4. Attrition evaluation",
         "Generated out-of-fold scores, classification metrics, calibration "
         "and lift charts, and High/Medium/Low risk bands.",
         "attrition_evaluation.md; metrics, band counts, charts"],
        ["5. Attendance anomalies",
         "Trained Isolation Forest without validation labels and compared model "
         "flags with feature-based rules on labelled examples.",
         "attendance-validation-report.md; metrics and type chart"],
    ]
    story.append(make_table(
        tasks, [23*mm, 79*mm, 70*mm], 7
    ))

    story.append(Paragraph("Task 1 — Data quality", styles["Section"]))
    story.append(Paragraph(
        "The attrition dataset contains 2,389 rows and 42 columns. There are "
        "2,383 unique employee IDs and six duplicate employee/date pairs. "
        "There are 349 recorded leavers (14.61%); the audit also identified "
        "213 invalid WFH values for review. The audit documented issues rather "
        "than silently treating questionable values as correct.",
        styles["SmallBody"]
    ))
    story.append(Paragraph("Task 2 — Feature pipeline", styles["Section"]))
    story.append(Paragraph(
        "The implementation separates feature preparation, preprocessing, and "
        "monthly attendance/payroll aggregation. The pipeline uses numeric and "
        "categorical preprocessing and excludes the target and identified "
        "leakage columns. The feature and pipeline test suite passed.",
        styles["SmallBody"]
    ))

    story.append(Paragraph("Task 3 — Baseline modelling", styles["Section"]))
    rows = [["Model", "ROC-AUC", "PR-AUC"]]
    for _, row in baseline.iterrows():
        name = str(row.get("model", row.iloc[0]))
        roc = row.get("roc_auc_mean", row.get("roc_auc", None))
        pr = row.get("pr_auc_mean", row.get("pr_auc", None))
        rows.append([name, fmt(roc), fmt(pr)])
    story.append(make_table(rows, [75*mm, 40*mm, 40*mm], 7.5))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "In the reported cross-validation results, Logistic Regression achieved "
        "ROC-AUC 0.7068 and PR-AUC 0.3176. The Dummy model provides a no-skill "
        "reference, while the depth-4 Decision Tree achieved ROC-AUC 0.6247. "
        "The benchmark comparison is saved separately for reproducibility.",
        styles["SmallBody"]
    ))

    story.append(PageBreak())

    # PAGE 2: Task 4
    story.append(Paragraph("Task 4 — Attrition metrics and risk bands", styles["Section"]))
    m = attr_metrics.iloc[0]
    metric_rows = [
        ["Metric", "Result", "Interpretation"],
        ["Accuracy", fmt(m["accuracy"]), "Overall classification accuracy"],
        ["Precision", fmt(m["precision"]), "Share of predicted leavers who left"],
        ["Recall", fmt(m["recall"]), "Share of actual leavers identified"],
        ["F1", fmt(m["f1"]), "Balance of precision and recall"],
        ["ROC-AUC", fmt(m["roc_auc"]), "Ranking discrimination"],
        ["PR-AUC", fmt(m["pr_auc"]), "Precision-recall performance"],
        ["Brier score", fmt(m["brier_score"]), "Probability error; lower is better"],
    ]
    story.append(make_table(metric_rows, [34*mm, 25*mm, 95*mm], 7.4))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "These are five-fold out-of-fold probabilities. At the 0.5 threshold, "
        "the model achieved recall 0.5989 and precision 0.2555. PR-AUC is useful "
        "because leavers are the minority class; accuracy alone can hide poor "
        "minority-class detection.",
        styles["SmallBody"]
    ))

    band_rows = [["Risk band", "Employees", "Leavers", "Observed leaver rate", "Share of leavers captured"]]
    for _, r in bands.iterrows():
        band_rows.append([
            str(r["risk_band"]), str(int(r["employee_count"])),
            str(int(r["actual_leavers"])),
            f'{float(r["observed_leaver_rate"]):.1%}',
            f'{float(r["share_of_all_leavers_captured"]):.1%}',
        ])
    story.append(Paragraph("Risk-band analysis", styles["Section"]))
    story.append(make_table(
        band_rows, [26*mm, 25*mm, 22*mm, 34*mm, 47*mm], 7
    ))
    story.append(Paragraph(
        "The High-risk band contains 239 employees, of whom 82 were actual "
        "leavers (34.31%). This is about 2.35 times the overall historical "
        "leaver rate of 14.61%. However, 154 actual leavers were in the Low "
        "band, so these bands should not be interpreted as guarantees.",
        styles["SmallBody"]
    ))
    add_image(story, "attrition_calibration_curve.png", 145*mm, 68*mm)
    story.append(Paragraph(
        "Calibration interpretation: points nearer the diagonal indicate closer "
        "agreement between predicted probabilities and observed rates. The chart "
        "should be interpreted cautiously because of finite sample sizes.",
        styles["Tiny"]
    ))
    add_image(story, "attrition_lift_chart.png", 125*mm, 48*mm)

    story.append(PageBreak())

    # PAGE 3: Task 5, testing, responsible use
    story.append(Paragraph("Task 5 — Attendance anomaly validation", styles["Section"]))
    a_model = anomaly[anomaly["method"] == "Isolation Forest only"].iloc[0]
    a_combined = anomaly[anomaly["method"] == "Isolation Forest + rules"].iloc[0]
    anomaly_rows = [
        ["Method", "Caught", "Missed", "Catch rate"],
        ["Isolation Forest only",
         str(int(a_model["caught"])), str(int(a_model["missed"])),
         f'{float(a_model["catch_rate"]):.1%}'],
        ["Isolation Forest + rules",
         str(int(a_combined["caught"])), str(int(a_combined["missed"])),
         f'{float(a_combined["catch_rate"]):.1%}'],
    ]
    story.append(make_table(
        anomaly_rows, [63*mm, 25*mm, 25*mm, 35*mm], 7.5
    ))
    story.append(Paragraph(
        "On 150 labelled anomaly examples, Isolation Forest caught 88 (58.67%) "
        "and missed 62. Adding simple attendance rules caught 111 (74.00%) and "
        "missed 39. The rules added 23 detections. The strongest remaining "
        "weakness was late-arrival spikes: only 3 of 30 were caught by either "
        "method. The type-level breakdown is available in the CSV evidence.",
        styles["SmallBody"]
    ))
    add_image(story, "attendance_validation_by_type.png", 150*mm, 73*mm)

    story.append(Paragraph("Testing and reproducibility", styles["Section"]))
    story.append(Paragraph(
        "The automated test suite completed with 5 passed tests. It covers "
        "feature engineering, model-ready output, target/leakage exclusion, "
        "pipeline save/load, and the error raised when transforming before fit. "
        "The project contains separate scripts and CSV/Markdown evidence for "
        "re-running the core evaluations.",
        styles["SmallBody"]
    ))

    story.append(Paragraph("Limitations and responsible use", styles["Section"]))
    story.append(Paragraph(
        "The attendance validation sheet contains labelled anomaly examples "
        "but not a matched set of confirmed normal records. Therefore, this "
        "evaluation measures catch rate on the supplied labelled examples but "
        "cannot establish precision or false-alarm rate. The workbook's baseline "
        "precision/recall values are reference values, not reproduced results. "
        "Attrition scores are estimates, not proof of future behaviour. They "
        "should support voluntary, human-reviewed support discussions, never "
        "stand-alone disciplinary or employment decisions. Further validation "
        "on independently reviewed data and fairness checks are needed.",
        styles["SmallBody"]
    ))

    story.append(Paragraph("Learning and next steps", styles["Section"]))
    story.append(Paragraph(
        "This phase strengthened skills in data auditing, feature preprocessing, "
        "cross-validation, probability-based evaluation, risk ranking, anomaly "
        "detection, testing, and evidence documentation. Next steps are to "
        "improve late-arrival anomaly rules, obtain a labelled normal validation "
        "set, assess calibration and fairness on future data, and package the "
        "pipeline for repeatable deployment.",
        styles["SmallBody"]
    ))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"PDF created: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
