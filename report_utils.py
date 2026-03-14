from typing import Optional


def build_warnings(report: dict, ext: str, r_bytes: bytes, docx_stats: Optional[dict]) -> list[str]:
    warns = []
    if ext == "pdf":
        warns.append("PDF detected. If text extraction seems incomplete, try saving as DOCX.")
    if docx_stats and docx_stats.get("approx_words", 0) > 1200:
        warns.append("Resume looks quite long; consider trimming to ~1 page (early career) or 2 pages (experienced).")
    if not report.get("action_verbs"):
        warns.append("Consider adding more strong action verbs (led, built, optimized, automated, …).")
    if report.get("contact_warnings"):
        warns.extend(report["contact_warnings"])
    return warns
