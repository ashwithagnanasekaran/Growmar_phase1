from pathlib import Path
import html
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "AIML_Ashwitha_Phase1_Report.pdf"

REPO_URL = "https://github.com/ashwithagnanasekaran/Growmar_phase1"
TAG_URL = REPO_URL + "/tree/phase-1"

# ---------- STYLES ----------

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#17324D"),
    spaceAfter=5,
))

styles.add(ParagraphStyle(
    name="SubTitle",
    parent=styles["Normal"],
    fontSize=9,
    leading=12.5,
    alignment=TA_CENTER,
    spaceAfter=7,
    splitLongWords=1,
))

styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=14,
    textColor=colors.HexColor("#17324D"),
    spaceBefore=5,
    spaceAfter=4,
    keepWithNext=True,
))

styles.add(ParagraphStyle(
    name="SmallBody",
    parent=styles["BodyText"],
    fontSize=8.8,
    leading=11.2,
    spaceAfter=4,
    splitLongWords=1,
))

styles.add(ParagraphStyle(
    name="Tiny",
    parent=styles["BodyText"],
    fontSize=7.6,
    leading=9.5,
    spaceAfter=2,
    splitLongWords=1,
))

styles.add(ParagraphStyle(
    name="TableCell",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.6,
    leading=9.2,
    spaceAfter=0,
    splitLongWords=1,
))

styles.add(ParagraphStyle(
    name="TableHead",
    parent=styles["BodyText"],
    fontName="Helvetica-Bold",
    fontSize=7.6,
    leading=9.2,
    textColor=colors.white,
    spaceAfter=0,
    splitLongWords=1,
))


# ---------- HELPERS ----------

def read_csv(filename):
    path = REPORTS / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required evidence file missing: {path}"
        )

    return pd.read_csv(path)


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def make_table(rows, widths):
    converted = []

    for row_index, row in enumerate(rows):
        converted_row = []

        for value in row:
            if isinstance(value, Paragraph):
                converted_row.append(value)
            else:
                text = html.escape(str(value))
                text = text.replace("\n", "<br/>")

                if row_index == 0:
                    style = styles["TableHead"]
                else:
                    style = styles["TableCell"]

                converted_row.append(
                    Paragraph(text, style)
                )

        converted.append(converted_row)

    table = Table(
        converted,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#243B53")),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("GRID", (0, 0), (-1, -1),
         0.35, colors.HexColor("#CBD5E1")),

        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),

        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    return table


def add_image(story, filename, max_width, max_height):
    path = REPORTS / filename

    if not path.exists():
        story.append(Paragraph(
            f"Chart not found: {html.escape(filename)}",
            styles["SmallBody"],
        ))
        return

    img = Image(str(path))

    scale = min(
        max_width / img.imageWidth,
        max_height / img.imageHeight,
    )

    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"

    story.append(img)
    story.append(Spacer(1, 2 * mm))


def footer(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748B"))

    canvas.drawString(
        17 * mm,
        9 * mm,
        "AIML Phase 1 | Evidence-based project report",
    )

    canvas.drawRightString(
        A4[0] - 17 * mm,
        9 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


# ---------- REPORT ----------

def main():

    baseline = read_csv("baseline_metrics.csv")
    read_csv("baseline_benchmark_comparison.csv")
    attr_metrics = read_csv("attrition_evaluation_metrics.csv")
    bands = read_csv("attrition_risk_band_counts.csv")
    anomaly = read_csv("attendance_validation_metrics.csv")

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=12 * mm,
        bottomMargin=15 * mm,
        title="AIML Ashwitha Phase 1 Report",
        author="AIML Phase 1 participant",
    )

    story = []

    # PAGE 1: OVERVIEW AND TASKS 1-3

    story.append(Paragraph(
        "AIML Phase 1 — Project Report",
        styles["ReportTitle"],
    ))

     
    
    repo_text = (
        "Name: Ashwitha | Track: AIML | Phase: 1<br/>"
        f"Repository: <link href='{REPO_URL}' color='#2563EB'>"
        "Open GitHub repository</link>"
        " | Git tag: "
        f"<link href='{TAG_URL}' color='#2563EB'>phase-1</link>"
    )

    story.append(Paragraph(repo_text, styles["SubTitle"]))

    story.append(Paragraph(
        "Executive summary",
        styles["Section"],
    ))

    story.append(Paragraph(
        "This phase covered data quality auditing, feature engineering, "
        "baseline attrition modelling, risk evaluation, and attendance-anomaly "
        "validation. The work includes reproducible scripts, automated tests, "
        "CSV evidence, charts, and written interpretations. Results are "
        "experimental and require independent validation before operational use.",
        styles["SmallBody"],
    ))

    story.append(Paragraph(
        "Task completion and evidence",
        styles["Section"],
    ))

    tasks = [
        ["Task", "Work completed", "Evidence"],
        [
            "1. Data audit",
            "Reviewed data quality, duplicate employee/date pairs, "
            "invalid values, target distribution, and planted issues.",
            "reports/data-audit.md; "
            "reports/data-audit-overview.png",
        ],
        [
            "2. Feature engineering",
            "Built preprocessing and monthly aggregation; tested "
            "model-ready features and leakage exclusions.",
            "src/features/; TASK2_EVIDENCE.md; tests/",
        ],
        [
            "3. Baseline modelling",
            "Compared Dummy, balanced Logistic Regression, and "
            "depth-4 Decision Tree with stratified five-fold validation.",
            "reports/baseline-report.md; baseline_metrics.csv; "
            "benchmark comparison",
        ],
        [
            "4. Attrition evaluation",
            "Generated out-of-fold scores, classification metrics, "
            "calibration/lift charts, and risk bands.",
            "reports/attrition_evaluation.md; metrics; charts",
        ],
        [
            "5. Attendance anomalies",
            "Evaluated Isolation Forest and attendance rules against "
            "labelled anomaly examples.",
            "reports/attendance-validation-report.md; "
            "metrics and type chart",
        ],
    ]

    story.append(make_table(
        tasks,
        [27 * mm, 79 * mm, 66 * mm],
    ))

    story.append(Paragraph(
        "Task 1 — Data quality",
        styles["Section"],
    ))

    story.append(Paragraph(
        "The attrition dataset contains 2,389 rows and 42 columns, "
        "with 2,383 unique employee IDs and six duplicate employee/date pairs. "
        "There are 349 recorded leavers (14.61%); 213 invalid WFH values "
        "were flagged for review. The audit documented questionable values "
        "rather than silently treating them as correct.",
        styles["SmallBody"],
    ))

    story.append(Paragraph(
        "Task 2 — Feature pipeline",
        styles["Section"],
    ))

    story.append(Paragraph(
        "The implementation separates feature preparation, preprocessing, "
        "and monthly attendance/payroll aggregation. Numeric and categorical "
        "preprocessing is used, and target/leakage columns are excluded. "
        "The feature and pipeline tests passed.",
        styles["SmallBody"],
    ))

    story.append(Paragraph(
        "Task 3 — Baseline modelling",
        styles["Section"],
    ))

    model_rows = [["Model", "ROC-AUC", "PR-AUC"]]

    for _, row in baseline.iterrows():
        name = str(row.get("model", row.iloc[0]))
        roc = row.get("roc_auc_mean", row.get("roc_auc", None))
        pr = row.get("pr_auc_mean", row.get("pr_auc", None))

        model_rows.append([
            name,
            fmt(roc),
            fmt(pr),
        ])

    story.append(make_table(
        model_rows,
        [82 * mm, 35 * mm, 35 * mm],
    ))

    story.append(Paragraph(
        "Logistic Regression achieved ROC-AUC 0.7068 and PR-AUC 0.3176 "
        "in the reported cross-validation results. The Dummy model is the "
        "no-skill reference; the depth-4 Decision Tree achieved ROC-AUC "
        "0.6247. Detailed benchmark comparison is available in the repository.",
        styles["SmallBody"],
    ))

    story.append(PageBreak())

    # PAGE 2: TASK 4

    story.append(Paragraph(
        "Task 4 — Attrition metrics and risk bands",
        styles["Section"],
    ))

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

    story.append(make_table(
        metric_rows,
        [31 * mm, 24 * mm, 107 * mm],
    ))

    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph(
        "These are five-fold out-of-fold probabilities. At threshold 0.5, "
        "recall was 0.5989 and precision was 0.2555. PR-AUC is useful because "
        "leavers are the minority class; accuracy alone can hide weak "
        "minority-class detection.",
        styles["SmallBody"],
    ))

    band_rows = [[
        "Risk band",
        "Employees",
        "Leavers",
        "Observed leaver rate",
        "Share of leavers captured",
    ]]

    for _, row in bands.iterrows():
        band_rows.append([
            str(row["risk_band"]),
            str(int(row["employee_count"])),
            str(int(row["actual_leavers"])),
            f'{float(row["observed_leaver_rate"]):.1%}',
            f'{float(row["share_of_all_leavers_captured"]):.1%}',
        ])

    story.append(Paragraph(
        "Risk-band analysis",
        styles["Section"],
    ))

    story.append(make_table(
        band_rows,
        [24 * mm, 23 * mm, 21 * mm, 38 * mm, 56 * mm],
    ))

    story.append(Paragraph(
        "The High-risk band contains 239 employees; 82 were actual leavers "
        "(34.31%), about 2.35 times the overall historical leaver rate of "
        "14.61%. However, 154 actual leavers were in the Low band, so risk "
        "bands are not guarantees.",
        styles["SmallBody"],
    ))

    add_image(
        story,
        "attrition_calibration_curve.png",
        124 * mm,
        53 * mm,
    )

    story.append(Paragraph(
        "Calibration: points nearer the diagonal indicate closer agreement "
        "between predicted probabilities and observed rates. Interpret "
        "cautiously because of finite sample sizes.",
        styles["Tiny"],
    ))

    add_image(
        story,
        "attrition_lift_chart.png",
        105 * mm,
        38 * mm,
    )

    story.append(PageBreak())

    # PAGE 3: TASK 5, TESTS, LIMITATIONS, LEARNING

    story.append(Paragraph(
        "Task 5 — Attendance anomaly validation",
        styles["Section"],
    ))

    a_model = anomaly[
        anomaly["method"] == "Isolation Forest only"
    ].iloc[0]

    a_combined = anomaly[
        anomaly["method"] == "Isolation Forest + rules"
    ].iloc[0]

    anomaly_rows = [
        ["Method", "Caught", "Missed", "Catch rate"],
        [
            "Isolation Forest only",
            str(int(a_model["caught"])),
            str(int(a_model["missed"])),
            f'{float(a_model["catch_rate"]):.1%}',
        ],
        [
            "Isolation Forest + rules",
            str(int(a_combined["caught"])),
            str(int(a_combined["missed"])),
            f'{float(a_combined["catch_rate"]):.1%}',
        ],
    ]

    story.append(make_table(
        anomaly_rows,
        [70 * mm, 23 * mm, 23 * mm, 35 * mm],
    ))

    story.append(Paragraph(
        "On 150 labelled anomaly examples, Isolation Forest caught 88 "
        "(58.67%) and missed 62. Adding attendance rules caught 111 (74.00%) "
        "and missed 39. The rules added 23 detections. Late-arrival spikes "
        "remain a weakness: only 3 of 30 were caught by either method. "
        "The type-level breakdown is in the CSV evidence.",
        styles["SmallBody"],
    ))

    add_image(
        story,
        "attendance_validation_by_type.png",
        137 * mm,
        61 * mm,
    )

    story.append(Paragraph(
        "Testing and reproducibility",
        styles["Section"],
    ))

    story.append(Paragraph(
        "The automated test suite completed with 5 passed tests. It covers "
        "feature engineering, model-ready output, target/leakage exclusion, "
        "pipeline save/load, and the error raised when transforming before fit. "
        "Scripts and CSV/Markdown evidence support re-running the evaluations.",
        styles["SmallBody"],
    ))

    story.append(Paragraph(
        "Limitations and responsible use",
        styles["Section"],
    ))

    story.append(Paragraph(
        "The attendance validation sheet contains labelled anomaly examples "
        "but no matched set of confirmed normal records, so catch rate can be "
        "measured but precision and false-alarm rate cannot. Workbook baseline "
        "precision/recall values are references, not reproduced results. "
        "Attrition scores are estimates, not proof of future behaviour. Use "
        "them only for voluntary, human-reviewed support discussions, never as "
        "stand-alone employment decisions. Further independent validation and "
        "fairness checks are needed.",
        styles["SmallBody"],
    ))

    story.append(Paragraph(
        "Learning and next steps",
        styles["Section"],
    ))

    story.append(Paragraph(
        "This phase strengthened skills in data auditing, preprocessing, "
        "cross-validation, probability-based evaluation, risk ranking, anomaly "
        "detection, testing, and evidence documentation. Next, I plan to improve "
        "late-arrival anomaly rules, obtain labelled normal examples, assess "
        "calibration and fairness on future data, and package the pipeline "
        "for repeatable deployment.",
        styles["SmallBody"],
    ))

    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    print(f"PDF created: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
