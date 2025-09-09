# ats_sections.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

# Canonical section keys used everywhere
SECTION_CANONICAL = (
    "summary", "experience", "projects", "education", "skills",
    "certifications", "publications"
)

# Section weights (used by frequency weighting etc.)
SECTION_WEIGHTS: Dict[str, float] = {
    "experience": 1.0, "work experience": 1.0, "professional experience": 1.0,
    "projects": 0.9, "summary": 0.8, "profile": 0.8,
    "education": 0.6, "skills": 0.6, "skills & tools": 0.6, "tech stack": 0.6,
}

# All headings we recognize (left-edge match)
_SECTION_PATTERNS = [
    # Experience family
    "experience", "work experience", "professional experience",
    # Projects / Education
    "projects", "education",
    # Skills family
    "skills", "skills & tools", "tech stack",
    # Extra tracked sections
    "certifications", "publications",
    # Summary family
    "summary", "professional summary", "profile summary", "profile", "objective",
    # Optional helpful variants
    "courses", "coursework", "activities", "extracurricular activities",
]

# Map variants → canonical
SECTION_ALIASES: Dict[str, str] = {
    # Experience
    "work experience": "experience",
    "professional experience": "experience",
    "experience": "experience",
    # Skills
    "skills & tools": "skills",
    "tech stack": "skills",
    "skills": "skills",
    # Summary
    "profile": "summary",
    "objective": "summary",
    "professional summary": "summary",
    "profile summary": "summary",
    "summary": "summary",
    # Direct canonicals
    "projects": "projects",
    "education": "education",
    "certifications": "certifications",
    "publications": "publications",
    # Optional variants
    "courses": "education",
    "coursework": "education",
    "activities": "projects",
    "extracurricular activities": "projects",
}

# Tolerant header patterns
_HEADER_START_RE = {
    name: re.compile(rf"^\s*{re.escape(name)}\b", re.I) for name in _SECTION_PATTERNS
}

def _normalize_line(s: str) -> str:
    # Lower + collapse inner whitespace
    return re.sub(r"\s+", " ", s.strip().lower())

def _try_match_header(candidates: List[str]) -> str | None:
    """Return matched raw name from _SECTION_PATTERNS, else None."""
    for cand in candidates:
        for name, rx in _HEADER_START_RE.items():
            if rx.match(cand):
                return name
    return None

def split_sections(text: str) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """
    Robust section splitter:
      - Recognizes a broad set of headings (case-insensitive).
      - Handles headings split across multiple visual lines (e.g., 'PROFESSIONAL' / 'EXPERIENCE').
      - Normalizes to canonical keys via SECTION_ALIASES.
      - Buckets content until the next header.

    Returns:
      sections_present: {canonical: bool}
      sections_text:    {canonical: "full section text"}
    """
    lines_raw = text.splitlines()
    lines_norm = [_normalize_line(ln) for ln in lines_raw]

    sections_present: Dict[str, bool] = {c: False for c in SECTION_CANONICAL}
    buckets: DefaultDict[str, List[str]] = defaultdict(list)

    current = "summary"  # default bucket for preamble
    i, n = 0, len(lines_norm)

    while i < n:
        l0_raw = lines_raw[i]
        l0 = lines_norm[i]

        # Build multi-line candidates (line, line+next, line+next+next2)
        cands = [l0] if l0 else []
        if i + 1 < n:
            cands.append(_normalize_line(lines_raw[i] + " " + lines_raw[i+1]))
        if i + 2 < n:
            cands.append(_normalize_line(lines_raw[i] + " " + lines_raw[i+1] + " " + lines_raw[i+2]))

        hit = _try_match_header(cands) if cands else None

        if hit:
            canon = SECTION_ALIASES.get(hit, hit)
            current = canon
            sections_present[canon] = True

            # Skip the lines consumed by the matched header
            if len(cands) >= 3 and _HEADER_START_RE[hit].match(cands[2]):
                i += 3
                continue
            if len(cands) >= 2 and _HEADER_START_RE[hit].match(cands[1]):
                i += 2
                continue
            i += 1
            continue

        # Not a header → add content to current bucket
        buckets[current].append(l0_raw)
        if l0.strip():
            sections_present[current] = True
        i += 1

    # Build compact text blocks
    sections_text: Dict[str, str] = {}
    for canon in SECTION_CANONICAL:
        block = "\n".join(buckets.get(canon, [])).strip()
        if block:
            sections_text[canon] = block

    return sections_present, sections_text
# Single-line header detector for other modules (e.g., to skip headers in bullet metrics)
def is_header_line(line: str) -> bool:
    s = re.sub(r"\s+", " ", line.strip().lower())
    # single-line check only; multi-line logic is handled in split_sections
    return any(rx.match(s) for rx in _HEADER_START_RE.values())
