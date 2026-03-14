import re

from fastapi import HTTPException


def _is_jd_placeholder(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in {"...", "..", ".", "n/a", "na", "none", "null", "nil", "test", "testing", "asdf", "qwerty", "jd"}:
        return True
    if re.fullmatch(r"[.\-_,;:!?/\\|]+", t):
        return True
    if re.fullmatch(r"\.{2,}", t):
        return True
    if re.fullmatch(r"(.)\1{4,}", t):
        return True
    return False


def validate_jd_hard(text: str):
    t = (text or "").strip()
    if len(t) < 15:
        raise HTTPException(400, "job_description is too short. Please provide at least 15 characters.")
    if _is_jd_placeholder(t):
        raise HTTPException(400, "job_description looks like placeholder text. Please provide a real job description.")


def assess_jd_quality(text: str) -> tuple[list[str], int, bool]:
    t = (text or "").strip()
    words = re.findall(r"[A-Za-z0-9+#/&.-]+", t.lower())
    wc = len(words)
    unique_ratio = (len(set(words)) / wc) if wc else 0.0
    signals = (
        "responsibilities",
        "requirements",
        "qualifications",
        "experience",
        "skills",
        "about the role",
        "must have",
        "nice to have",
        "about us",
        "job description",
    )
    signal_hits = sum(1 for s in signals if s in t.lower())

    warnings = []
    score = 100
    if wc < 60:
        warnings.append("Job description is short; score reliability may be lower.")
        score -= 25
    if signal_hits == 0:
        warnings.append("Could not detect common JD sections (requirements/responsibilities); results may be less reliable.")
        score -= 20
    if wc >= 20 and unique_ratio < 0.35:
        warnings.append("Job description has low keyword diversity; avoid repeated generic words.")
        score -= 20
    if wc < 25:
        score -= 20
    if len(t) < 50:
        score -= 15

    score = max(0, min(100, score))
    low_confidence = score < 70
    return warnings, score, low_confidence
