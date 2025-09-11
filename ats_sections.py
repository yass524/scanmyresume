# ats_sections.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
# ---- Section canonicalization (keep one truth for each family) ----
_SECTION_FAMILIES = {
    "experience": {"experience", "work experience", "professional experience"},
    "skills": {"skills", "skills & tools", "tech stack"},
    "summary": {"summary", "professional summary", "objective", "profile", "profile summary"},
    # singletons not needed here: "projects", "education", etc.
}

def _canonicalize_sections(sp: Dict[str, bool]) -> Dict[str, bool]:
    sp = dict(sp or {})
    out: Dict[str, bool] = {}

    # families: set canonical True if ANY alias is True
    for canon, aliases in _SECTION_FAMILIES.items():
        out[canon] = any(sp.get(a, False) for a in aliases)

    # pass through common singletons (don’t mark missing; just reflect whatever the splitter found)
    for k in ("projects", "education", "certifications", "publications"):
        out[k] = bool(sp.get(k, False))

    # keep originals too (optional): mirror the canonical truth so they don’t show as “No”
    # (This prevents “work experience: No” when “experience: Yes”.)
    for canon, aliases in _SECTION_FAMILIES.items():
        v = out[canon]
        for a in aliases:
            out[a] = v

    return out

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
    "experience", "work experience", "professional experience", "employment", "career history", "work history",
    # Projects / Education
    "projects", "selected projects", "academic projects", "education", "academics", "qualifications",
    # Skills family
    "skills", "skills & tools", "tech stack", "technical skills", "core competencies", "tooling",
    # Extra tracked sections
    "certifications", "licenses", "publications",
    # Summary family
    "summary", "professional summary", "profile summary", "profile", "objective", "about",
    # Optional helpful variants
    "courses", "coursework", "activities", "extracurricular activities", "awards", "honors", "achievements",
]

# Map variants → canonical
# Map variants → canonical
SECTION_ALIASES: Dict[str, str] = {
    # Experience family
    "work experience": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "career history": "experience",
    "work history": "experience",
    "experience": "experience",

    # Skills family
    "skills & tools": "skills",
    "tech stack": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "tooling": "skills",
    "skills": "skills",

    # Summary family
    "profile": "summary",
    "objective": "summary",
    "professional summary": "summary",
    "profile summary": "summary",
    "about": "summary",
    "summary": "summary",

    # Direct canonicals
    "projects": "projects",
    "selected projects": "projects",
    "academic projects": "projects",
    "education": "education",
    "academics": "education",
    "qualifications": "education",
    "certifications": "certifications",
    "licenses": "certifications",
    "publications": "publications",

    # Optional variants
    # NOTE: “courses” often stands in for certs—count it as BOTH:
    "courses": "certifications",      # ← new
    "coursework": "education",
    "activities": "projects",
    "extracurricular activities": "projects",
    "awards": "projects",
    "honors": "projects",
    "achievements": "projects",
}

# Tolerant header patterns (left-edge, case-insensitive)
_HEADER_START_RE = {
    name: re.compile(rf"^\s*{re.escape(name)}\b", re.I) for name in _SECTION_PATTERNS
}

# General "looks like a header" shape (ALL-CAPS or Title Case + optional separators)
_HEADER_SHAPE = re.compile(
    r""" ^
        (?:
          [A-Z][A-Z\s/&\-|]+                 # ALL-CAPS blocks
          |
          [A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4} # Title Case (≤5 words)
        )
        \s*[:\|\-\–\—]*\s*$                  # optional trailing separators
    """, re.X
)

def _normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _try_match_header(candidates: List[str]) -> str | None:
    """Return matched raw name from _SECTION_PATTERNS, else None."""
    for cand in candidates:
        for name, rx in _HEADER_START_RE.items():
            if rx.match(cand):
                return name
    return None

# ---------- NEW: two-column splitter (handles left sidebars) ----------
def _split_columns(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    If many lines contain long whitespace gaps, assume two-column layout.
    Split into (left, right). Otherwise return ([], lines).
    """
    gaps = []
    for ln in lines:
        m = re.search(r"(\s{6,}|\t{2,})", ln)
        if m:
            gaps.append(m.start())
    if len(gaps) < max(6, len(lines)//12):
        return [], lines

    cut = sorted(gaps)[len(gaps)//2]
    left, right = [], []
    for ln in lines:
        if len(ln) > cut and ln[cut].isspace():
            left.append(ln[:cut].rstrip())
            right.append(ln[cut:].lstrip())
        else:
            right.append(ln.rstrip())

    left  = [l for l in (l.strip() for l in left)  if l]
    right = [r for r in (r.strip() for r in right) if r]
    return left, right

def _segment_stream(lines_raw: List[str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    lines_norm = [_normalize_line(ln) for ln in lines_raw]

    sections_present: Dict[str, bool] = {c: False for c in SECTION_CANONICAL}
    buckets: DefaultDict[str, List[str]] = defaultdict(list)

    current_targets: list[str] = ["summary"]   # one or more active sections
    i, n = 0, len(lines_norm)

    while i < n:
        l0_raw = lines_raw[i]
        l0 = lines_norm[i]
        cands = [l0] if l0 else []
        if i + 1 < n: cands.append(_normalize_line(lines_raw[i] + " " + lines_raw[i+1]))
        if i + 2 < n: cands.append(_normalize_line(lines_raw[i] + " " + lines_raw[i+1] + " " + lines_raw[i+2]))

        # strict name hit from patterns?
        hit_name = None
        for cand in cands:
            if not cand: continue
            for name, rx in _HEADER_START_RE.items():
                if rx.match(cand):
                    hit_name = name; break
            if hit_name: break

        canon_list: list[str] = []
        if hit_name:
            canon_list = [SECTION_ALIASES.get(hit_name, hit_name)]
        else:
            # shape fallback (ALL-CAPS sidebar etc.)
            if l0 and _HEADER_SHAPE.match(lines_raw[i].strip()):
                canon_list = _canonicalize_guess(lines_raw[i])

        if canon_list:
            # mark presence for ALL implied sections
            for c in canon_list:
                if c in sections_present:
                    sections_present[c] = True
            current_targets = canon_list  # route following lines to all of them

            # skip consumed lines (prefer longest match)
            if len(cands) >= 3 and hit_name and _HEADER_START_RE[hit_name].match(cands[2]): i += 3; continue
            if len(cands) >= 2 and hit_name and _HEADER_START_RE[hit_name].match(cands[1]): i += 2; continue
            i += 1
            continue

        # Not a header → add content to every active target
        for tgt in current_targets:
            buckets[tgt].append(l0_raw)
            if l0.strip():
                sections_present[tgt] = True
        i += 1

    # compact blocks
    sections_text: Dict[str, str] = {}
    for canon in SECTION_CANONICAL:
        block = "\n".join(buckets.get(canon, [])).strip()
        if block:
            sections_text[canon] = block

    return sections_present, sections_text

def _canonicalize_guess(line: str) -> list[str]:
    """Return 0+ canonical names that this header line implies (supports combined headers)."""
    s = _normalize_line(re.sub(r"[:\|\-\–\—]+$", "", line).strip())
    # split on common joiners: "projects & skills", "projects and skills"
    parts = re.split(r"\s*(?:&|/|,|and)\s*", s)
    out = []
    for p in parts:
        # exact alias
        if p in SECTION_ALIASES:
            out.append(SECTION_ALIASES[p])
        else:
            # startswith heuristic (e.g., "selected projects")
            for key in SECTION_ALIASES:
                if p.startswith(key):
                    out.append(SECTION_ALIASES[key])
                    break
    # special case: “courses” should also count toward education presence
    if "courses" in s and "education" not in out:
        out.append("education")
    # de-dup and keep canonical order preference
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

# ---------- Public API ----------
def split_sections(text: str) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """
    Robust section splitter with two-column support and combined headers.
    Returns:
      sections_present: {canonical: bool}
      sections_text:    {canonical: "full section text"}
    """
    raw_lines = text.splitlines()
    left, right = _split_columns(raw_lines)

    present: Dict[str, bool] = {c: False for c in SECTION_CANONICAL}
    text_map: Dict[str, str] = {}

    # parse helper
    def parse_stream(lines: List[str]):
        p, t = _segment_stream(lines)
        for k in present:
            present[k] = present[k] or p.get(k, False)
        for k, v in t.items():
            if k in text_map:
                text_map[k] = (text_map[k] + "\n" + v).strip()
            else:
                text_map[k] = v

    if left:
        parse_stream(left)
    parse_stream(right if right else raw_lines)

    return present, text_map

# Single-line header detector for other modules (e.g., to skip headers in bullet metrics)
def is_header_line(line: str) -> bool:
    s = re.sub(r"\s+", " ", line.strip().lower())
    if any(rx.match(s) for rx in _HEADER_START_RE.values()):
        return True
    return bool(_HEADER_SHAPE.match(line.strip()))
def canonicalize_sections(sp: Dict[str, bool]) -> Dict[str, bool]:
    return _canonicalize_sections(sp)
