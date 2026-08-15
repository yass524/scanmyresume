from pathlib import Path

from fastapi import HTTPException

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    PDF_OK = True
except Exception:
    PDF_OK = False


def build_pdf_from_report(report: dict, out_path: Path):
    if not PDF_OK:
        raise HTTPException(500, "PDF export not available. Install 'reportlab'.")

    styles = getSampleStyleSheet()
    story = []

    title = f"ATS Report • Score: {report.get('score', 0)}/100"
    story.append(Paragraph(title, styles["Title"]))
    if report.get("rule_based_score") is not None and report.get("rule_based_score") != report.get("score"):
        story.append(Paragraph(f"Rule-based score: {report.get('rule_based_score')}/100", styles["BodyText"]))
    story.append(Spacer(1, 12))

    comps = report.get("components", {})
    if comps:
        data = [["Component", "Value"]] + [[k.replace("_", " "), str(v)] for k, v in comps.items()]
        t = Table(data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#111827"), colors.HexColor("#0b1220")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(Paragraph("Components", styles["Heading2"]))
        story.append(t)
        story.append(Spacer(1, 12))

    def para_list(title: str, arr: list[str]):
        story.append(Paragraph(title, styles["Heading2"]))
        if arr:
            story.append(Paragraph(", ".join(arr), styles["BodyText"]))
        else:
            story.append(Paragraph("None", styles["BodyText"]))
        story.append(Spacer(1, 8))

    para_list("Matched Keywords", report.get("matched_keywords") or [])
    para_list("Missing Keywords", report.get("missing_keywords") or [])
    para_list("AI Matched Terms", report.get("ai_matched_terms") or [])

    ai = report.get("ai_feedback") or {}
    if ai.get("used"):
        story.append(Paragraph("AI Feedback", styles["Heading2"]))
        story.append(Paragraph(str(ai.get("summary") or ""), styles["BodyText"]))
        story.append(Spacer(1, 6))
        para_list("AI Strengths", ai.get("strengths") or [])
        para_list("AI Improvements", ai.get("improvements") or [])

    sugg = (ai.get("rewritten_bullets") if ai.get("used") else None) or report.get("suggested_bullets") or []
    story.append(Paragraph("Suggested Bullets", styles["Heading2"]))
    if sugg:
        for s in sugg:
            story.append(Paragraph(f"• {s}", styles["BodyText"]))
    else:
        story.append(Paragraph("None", styles["BodyText"]))
    story.append(Spacer(1, 12))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    doc.build(story)
