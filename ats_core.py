# ats_core.py

from __future__ import annotations
import os, re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Iterable

# Compiled skill maps / alias regexes
from ats_skills import ALIAS_RE as _ALIAS_RE
from ats_skills import CANON_CATEGORY as _CANON_CATEGORY
from ats_skills import CANON_ALIASES as _CANON_ALIASES

# Robust section splitter + helpers
from ats_sections import split_sections as _split_sections
from ats_sections import SECTION_WEIGHTS
from ats_sections import is_header_line as _is_header_line

# Lexicon (verbs, stopwords, common regexes, markers)
from ats_lexicon import (
    ACTION_VERBS,
    STOPWORDS,
    RE_DIGIT,
    RE_CURRENCY,
    RE_EMAIL,
    RE_PHONE,
    RE_PCT,
    RE_URL,
    BULLET_MARKERS,
)

# -----------------------------
# Config / thresholds (env-tunable)
# -----------------------------
EMB_ON: bool = os.environ.get("EMB_ON", "0") == "1"
EMB_MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMB_THRESH: float = float(os.environ.get("EMB_THRESH", "0.60"))

# Score weights
W_REQUIRED_COVERAGE = 0.50
W_OVERALL_COVERAGE  = 0.20
W_FREQ_REQ          = 0.15
W_FREQ_ALL          = 0.05
W_SECTION_HYGIENE   = 0.10

# Penalties (relaxed for short, dynamic for stuffing)
MIN_WORDS = 80
MAX_WORDS = 1500
PENALTY_LENGTH = 5.0

# Frequency caps (diminishing returns)
FREQ_CAP_REQUIRED = 4
FREQ_CAP_OVERALL  = 6

# ----------------- Embeddings (optional) -----------------
_emb_model = None
_emb_cache: Dict[str, List[float]] = {}

def _maybe_load_model():
    """Load sentence-transformers model lazily if EMB_ON=1."""
    global _emb_model
    if not EMB_ON or _emb_model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    except Exception:
        _emb_model = None

def _embed(texts: List[str]) -> List[List[float]]:
    _maybe_load_model()
    if _emb_model is None:
        return [[0.0] for _ in texts]
    out = []
    for t in texts:
        vec = _emb_cache.get(t)
        if vec is None:
            vec = _emb_model.encode([t], normalize_embeddings=True)[0].tolist()
            _emb_cache[t] = vec
        out.append(vec)
    return out

def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def _vec_scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]

def _cos(a: List[float], b: List[float]) -> float:
    # inputs are normalized
    return sum(x * y for x, y in zip(a, b))

# ----------------- Text utils -----------------
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\r\t]", " ", s)
    s = re.sub(r"[^\w\.\-\+\&/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sentences(text: str) -> List[str]:
    return re.split(r"[.!?\n]+", text)

def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9\+\#/&.-]+", text.lower()) if w not in STOPWORDS]

def _ngrams(words: List[str], n: int) -> Iterable[str]:
    for i in range(len(words) - n + 1):
        yield " ".join(words[i : i + n])

# ----------------- Bullet / verbs -----------------
def _bullet_and_verb_metrics(text: str) -> Tuple[float, float]:
    """
    Returns (bullet_ratio_pct, verb_ratio_pct), both in PERCENT scale.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    content_lines = [ln for ln in lines if not _is_header_line(ln)]
    if not content_lines:
        return 0.0, 0.0

    bullets = [ln for ln in content_lines if re.match(r"^[\-\u2022\*\•·]", ln)]
    bullet_ratio_pct = 100.0 * len(bullets) / max(1, len(content_lines))

    verb_starts = 0
    for ln in bullets:
        tokens = re.findall(r"[a-zA-Z]+", ln.lower())
        if tokens and tokens[0] in ACTION_VERBS:
            verb_starts += 1
    verb_ratio_pct = 100.0 * verb_starts / max(1, len(bullets)) if bullets else 0.0

    return round(bullet_ratio_pct, 1), round(verb_ratio_pct, 1)

# ----------------- JD parsing -----------------
_REQ_CUES  = ("requirements", "required", "must have", "must-have", "qualifications", "you have")
_PREF_CUES = ("preferred", "nice to have", "nice-to-have", "bonus", "plus", "good to have")

def _jd_required_preferred(jd_text: str) -> Tuple[str, str]:
    text = jd_text.lower()
    pref_idx = min([text.find(k) for k in _PREF_CUES if k in text] or [len(text) + 1])
    if pref_idx != len(text) + 1 and pref_idx > 0:
        return jd_text[:pref_idx].strip(), jd_text[pref_idx:].strip()
    return jd_text.strip(), ""

# ----------------- Skill extraction -----------------
def _extract_skills(text: str) -> Tuple[Counter, Set[str]]:
    matches: Counter = Counter()
    matched: Set[str] = set()
    for alias, (canonical, pat) in _ALIAS_RE.items():
        hits = len(re.findall(pat, text))
        if hits:
            matches[canonical] += hits
            matched.add(canonical)
    return matches, matched

def _weighted_counts_by_section(resume_text: str, sections_text: dict[str, str] | None = None) -> Counter:
    """
    Count matched skills per section with weights, using parsed section text if provided
    to avoid re-splitting.
    """
    if sections_text is None:
        _, sections_text = _split_sections(resume_text)

    weighted = Counter()
    for sec, text in sections_text.items():
        weight = SECTION_WEIGHTS.get(sec, 0.7)
        counts, _ = _extract_skills(_norm(text))
        for k, v in counts.items():
            weighted[k] += v * weight
    return weighted

# ----------------- Semantic credit -----------------
def _canon_aliases(canon: str) -> List[str]:
    return sorted(_CANON_ALIASES.get(canon, {canon}), key=len, reverse=True)

def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return [0.0]
    acc = [0.0] * len(vectors[0])
    for v in vectors:
        acc = _vec_add(acc, v)
    return _vec_scale(acc, 1.0 / len(vectors))

def _semantic_credit(missing: List[str], resume_text: str) -> Tuple[List[str], Dict[str, float]]:
    if not EMB_ON:
        return [], {}
    _maybe_load_model()
    if _emb_model is None:
        return [], {}

    words = _words(resume_text)
    cands = set()
    for n in (1, 2, 3):
        for ng in _ngrams(words, n):
            if any(tok in STOPWORDS for tok in ng.split()):
                continue
            cands.add(ng)
    if not cands:
        return [], {}

    cand_list = sorted(cands)
    cand_vecs = _embed(cand_list)

    jd_vecs = []
    for cname in missing:
        aliases = _canon_aliases(cname)
        alias_vecs = _embed(aliases)
        jd_vecs.append(_centroid(alias_vecs))

    ai_hits = []
    sims: Dict[str, float] = {}
    for i, cname in enumerate(missing):
        a = jd_vecs[i]
        best = 0.0
        for j in range(len(cand_list)):
            s = _cos(a, cand_vecs[j])
            if s > best:
                best = s
        if best >= EMB_THRESH:
            ai_hits.append(cname)
            sims[cname] = round(float(best), 3)
    return ai_hits, sims

# ----------------- Coverage & frequency -----------------
def _coverage(required: Set[str], preferred: Set[str], resume_matches: Set[str], ai_hits: Set[str]) -> Tuple[float, float, List[str], List[str]]:
    matched = resume_matches | ai_hits
    req_cov = 100.0 * (len(required & matched) / max(1, len(required))) if required else 100.0
    overall_set = required | preferred
    overall_cov = 100.0 * (len(overall_set & matched) / max(1, len(overall_set))) if overall_set else 100.0
    req_missing = sorted(list(required - matched))
    pref_missing = sorted(list(preferred - matched))
    return round(req_cov, 1), round(overall_cov, 1), req_missing, pref_missing

def _freq_bonus(counts: Counter, targets: Set[str], cap: int) -> float:
    if not targets:
        return 0.0
    s = 0.0
    for t in targets:
        c = counts.get(t, 0.0)
        s += min(c, cap) / cap
    return round(100.0 * s / len(targets), 1)

# ----------------- Penalties -----------------
def _length_penalty(words_count: int) -> bool:
    return words_count < MIN_WORDS or words_count > MAX_WORDS

def _stuffing_penalty_amount(counts: Counter, total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    max_c = max(counts.values() or [0.0])
    ratio = max_c / max(1.0, total_words)

    dense = any((c >= 14) for c in counts.values())
    if not dense and ratio < 0.30:
        return 0.0
    if ratio >= 0.60 or max_c >= 40:
        return 26.0
    if ratio >= 0.45 or max_c >= 25:
        return 18.0
    return 12.0

# ----------------- Suggestion heuristics (PERCENT scale) -----------------
MIN_BULLET_RATIO_PCT = 25.0   # ≥25% of lines should be bullets
MIN_VERB_RATIO_PCT   = 15.0   # ≥15% of bullets start with action verbs
MIN_NUMERIC_DENSITY  = 0.010  # fraction of numeric-like tokens
MAX_PAGES_EARLY      = 1.0
MAX_PAGES_GENERAL    = 2.0
MIN_CONTACT_FIELDS   = 2
MIN_LINKS            = 1
MIN_SKILLS_COUNT     = 6
MIN_EDU_RECENCY_YRS  = 3
MAX_LINE_LEN_CHARS   = 180
MAX_SOFT_SUGGESTIONS = 3
SUGGESTIONS_MAX_OUT  = 14

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%$€£]+", text.lower())

def _numeric_density(text: str) -> float:
    toks = _tokenize(text)
    if not toks:
        return 0.0
    hits = 0
    for t in toks:
        if RE_DIGIT.search(t) or RE_PCT.search(t) or RE_CURRENCY.search(t):
            hits += 1
    return hits / max(1, len(toks))

def _bullet_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln[:1] in BULLET_MARKERS or (len(ln) >= 2 and ln[:2] in {"- ", "• "})]

def _lines(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]

def _bullet_ratio(text: str) -> float:
    """Return fraction (0..1) of lines that are bullets."""
    L = _lines(text)
    B = _bullet_lines(text)
    return (len(B) / max(1, len(L))) if L else 0.0

def _verb_ratio(text: str) -> float:
    """Return fraction (0..1) of bullet lines that start with an action verb."""
    B = _bullet_lines(text)
    if not B:
        return 0.0
    good = 0
    for ln in B:
        m = re.match(r"^[•\-\–\—\*]?\s*([A-Za-z]+)", ln)
        if not m:
            continue
        first = m.group(1).lower()
        if first in ACTION_VERBS:
            good += 1
    return good / max(1, len(B))

def _has_contact(text: str) -> int:
    count = 0
    if RE_EMAIL.search(text): count += 1
    if RE_PHONE.search(text): count += 1
    if re.search(r"linkedin\.com|github\.com|behance\.net|dribbble\.com|gitlab\.com", text, re.I):
        count += 1
    return count

def _links_count(text: str) -> int:
    return len(RE_URL.findall(text))

def _long_lines_exist(text: str, max_len: int = MAX_LINE_LEN_CHARS) -> bool:
    return any(len(ln) > max_len for ln in _lines(text))

def _has_tables_or_columns_hint(text: str) -> bool:
    # crude: tabs or multiple spacing alignment
    return ("\t" in text) or bool(re.search(r"[^\S\r\n]{3,}\S+[^\S\r\n]{3,}", text))

def _skills_list_count(sections: dict[str, str]) -> int:
    skills_block = ""
    for key in ("skills", "skills & tools", "tech stack"):
        if key in sections:
            skills_block += "\n" + sections[key]
    if not skills_block and "resume_text" in sections:
        skills_block = sections["resume_text"]
    toks = _tokenize(skills_block)
    commas = skills_block.count(",")
    bullets = len(_bullet_lines(skills_block))
    return max(commas + 1, bullets * 3, len({t for t in toks if len(t) > 1}) // 12)

def _recent_grad(year: int | None, now_year: int) -> bool:
    return (year is not None) and (now_year - year <= MIN_EDU_RECENCY_YRS)

def _first_person_ratio(text: str) -> float:
    toks = _tokenize(text)
    if not toks:
        return 0.0
    fp = sum(t in {"i", "my", "me", "mine"} for t in toks)
    return fp / len(toks)

def _passive_voice_ratio(text: str) -> float:
    matches = re.findall(r"\b(was|were|been|being)\b\s+\w+\b\s+\bby\b", text.lower())
    toks = _tokenize(text)
    return len(matches) / max(1, len(toks))

from time import localtime

def _suggestions(
    req_missing: list[str],
    pref_missing: list[str],
    resume_text: str,
    sections: dict[str, str] | None = None,
    page_count: float | None = None,
    job_title: str | None = None,
    is_early_career: bool | None = None,
    grad_year: int | None = None,
    bullet_ratio: float | None = None,   # may be fraction or percent
    verb_ratio: float | None = None,     # may be fraction or percent
) -> list[str]:
    """
    Returns prioritized suggestions with conditions.
    `sections` may include a special key "resume_text" for fallback analyzers.
    """
    sections = sections or {}

    # Normalize bullet/verb ratios to PERCENT scale for comparison
    if bullet_ratio is None:
        bullet_ratio_pct = 100.0 * _bullet_ratio(resume_text)   # _bullet_ratio returns fraction
    else:
        bullet_ratio_pct = bullet_ratio * 100.0 if bullet_ratio <= 1.0 else bullet_ratio

    if verb_ratio is None:
        verb_ratio_pct = 100.0 * _verb_ratio(resume_text)       # _verb_ratio returns fraction
    else:
        verb_ratio_pct = verb_ratio * 100.0 if verb_ratio <= 1.0 else verb_ratio

    numeric_density = _numeric_density(resume_text)
    contact_fields  = _has_contact(resume_text)
    links_cnt       = _links_count(resume_text)
    long_lines      = _long_lines_exist(resume_text)
    tables_hint     = _has_tables_or_columns_hint(resume_text)
    skills_count    = _skills_list_count({**sections, "resume_text": resume_text})
    first_person    = _first_person_ratio(resume_text)
    passive_ratio   = _passive_voice_ratio(resume_text)

    now_year        = localtime().tm_year
    is_recent_grad  = _recent_grad(grad_year, now_year) if grad_year is not None else False

    S: list[tuple[int, str]] = []

    # 1) Hard requirements & preferences
    for m in req_missing[:8]:
        S.append((0, f"Add a concrete achievement using '{m}' (e.g., “Implemented {m} to improve a KPI by X%”)."))
    for m in pref_missing[:5]:
        S.append((2, f"If relevant, mention '{m}' with one concise bullet or in Skills."))

    # 2) Evidence/impact quality
    if numeric_density < MIN_NUMERIC_DENSITY:
        S.append((1, "Quantify achievements (e.g., “Reduced downtime by 15%”, “Saved $50K/year”)."))

    # 3) Structure & readability
    if bullet_ratio_pct < MIN_BULLET_RATIO_PCT:
        S.append((2, "Use bullets for experience; target ≥25% of lines as bullets."))
    if verb_ratio_pct < MIN_VERB_RATIO_PCT:
        S.append((2, "Start bullets with strong action verbs (Led, Built, Optimized, Spearheaded…)."))
    if long_lines:
        S.append((3, "Split long lines into concise bullets (~1–2 lines each) to improve readability."))
    if tables_hint:
        S.append((3, "Avoid multi-column layouts/tables; use a single-column, ATS-friendly format."))

    # 4) Contact & links
    if contact_fields < MIN_CONTACT_FIELDS:
        S.append((1, "Include at least email and phone (and optionally city/country)."))
    if links_cnt < MIN_LINKS:
        if job_title and any(k in (job_title or "").lower() for k in ("engineer","developer","data","ml","ai","software","frontend","backend","fullstack")):
            S.append((2, "Add a professional link (GitHub/LinkedIn/Portfolio) to showcase projects."))
        elif job_title and any(k in (job_title or "").lower() for k in ("designer","ux","ui","product")):
            S.append((2, "Add a portfolio link (Behance/Dribbble/Website) to showcase work."))
        else:
            S.append((2, "Add a professional link (LinkedIn, portfolio, or GitHub) for credibility."))

    # 5) Sections & ordering
    has_exp  = any(k in sections for k in ("experience", "work experience", "professional experience"))
    has_proj = "projects" in sections
    if not (has_exp or has_proj):
        S.append((1, "Add an 'Experience' or 'Projects' section highlighting relevant work."))
    if has_proj and not has_exp and is_recent_grad:
        S.append((3, "Place 'Projects' above 'Education' to emphasize hands-on experience."))

    # 6) Skills list hygiene
    if skills_count < MIN_SKILLS_COUNT:
        S.append((2, "Expand the Skills section with role-relevant tools (keep to keywords; avoid sentences)."))

    # 7) Tailoring signals
    if req_missing or pref_missing:
        S.append((1, "Tailor keywords to the job description (mirror terminology where appropriate)."))

    # 8) Length & recency
    if page_count is not None:
        if is_early_career and page_count > MAX_PAGES_EARLY:
            S.append((2, "Limit to 1 page for early-career resumes; keep only the most relevant items."))
        elif (not is_early_career) and page_count > MAX_PAGES_GENERAL:
            S.append((3, "Keep to 2 pages max; trim older or less relevant entries."))

    # 9) Students / recent grads
    if is_recent_grad:
        S.append((3, "Include key coursework, GPA (if strong), and top projects relevant to the role."))

    # 10) Writing style checks
    if first_person > 0.01:
        S.append((3, "Avoid first-person pronouns in bullets (write “Built X…” not “I built X…”)."))
    if passive_ratio > 0.002:
        S.append((3, "Prefer active voice (“Automated X”) over passive (“X was automated by…”)."))

    # 11) Soft skills (cap how many)
    soft_added = 0
    if soft_added < MAX_SOFT_SUGGESTIONS and passive_ratio > 0.002:
        S.append((4, "Demonstrate soft skills via outcomes (e.g., “Led a team of 4 to deliver on time”).")); soft_added += 1
    if soft_added < MAX_SOFT_SUGGESTIONS and numeric_density < MIN_NUMERIC_DENSITY:
        S.append((4, "Use the STAR method in bullets (Situation, Task, Action, Result).")); soft_added += 1
    if soft_added < MAX_SOFT_SUGGESTIONS and (has_exp or has_proj):
        S.append((4, "Balance soft skills with evidence (teamwork, leadership) tied to measurable results.")); soft_added += 1

    # 12) File/export hygiene
    if "resume.doc" in resume_text.lower() or ".docx" in resume_text.lower():
        S.append((4, "Export as PDF (text-based, not scanned) to keep formatting and remain ATS-friendly."))

    # De-duplicate while preserving best priority
    seen = {}
    for pr, txt in S:
        if txt not in seen or pr < seen[txt]:
            seen[txt] = pr
    ranked = sorted(((pr, txt) for txt, pr in seen.items()), key=lambda x: x[0])
    return [txt for _, txt in ranked[:SUGGESTIONS_MAX_OUT]]

# ----------------- Public API -----------------
def compute_score(resume_text: str, job_description: str) -> Dict:
    rt = resume_text or ""
    jt = job_description or ""
    if not rt.strip() or not jt.strip():
        return {"score": 0.0, "error": "Missing resume or job description"}

    rt_norm = _norm(rt)
    jt_norm = _norm(jt)

    # Sections (present + text), canonicalized presence for report/UI
    # Sections (present + text) — already canonical from ats_sections
    sections_presence, sections_text = _split_sections(rt)  # pass RAW text (not normalized)

    # Bullet / verbs (percent scale)
    bullet_ratio_pct, verb_ratio_pct = _bullet_and_verb_metrics(rt)

    # JD: split into required vs preferred (best-effort)
    jd_req_txt, jd_pref_txt = _jd_required_preferred(jt.lower())
    req_counts, _  = _extract_skills(_norm(jd_req_txt))
    pref_counts, _ = _extract_skills(_norm(jd_pref_txt))
    jd_required  = set(req_counts.keys()) if req_counts else set()
    jd_preferred = set(pref_counts.keys()) if pref_counts else set()
    if not jd_required and not jd_preferred:
        all_jd_counts, _ = _extract_skills(jt_norm)
        jd_required = set(all_jd_counts.keys())

    # Coverage (initial, without semantic credit)
    _, resume_matched = _extract_skills(rt_norm)
    req_cov0, overall_cov0, req_missing0, pref_missing0 = _coverage(jd_required, jd_preferred, resume_matched, set())

    # Optional semantic credit
    ai_on = bool(EMB_ON)
    ai_matches_list: List[str] = []
    ai_sim_map: Dict[str, float] = {}
    if EMB_ON:
        missing_all = sorted(set(req_missing0 + pref_missing0))
        ai_hits, sims = _semantic_credit(missing_all, rt_norm)
        ai_matches_list = sorted(ai_hits)
        ai_sim_map = sims

    # Recompute coverage with semantic credit
    req_cov, overall_cov, req_missing, pref_missing = _coverage(
        jd_required, jd_preferred, resume_matched, set(ai_matches_list)
    )

    # Frequency bonuses (weighted by section)
    resume_counts_weighted = _weighted_counts_by_section(rt, sections_text)
    freq_req = _freq_bonus(resume_counts_weighted, jd_required - set(req_missing), FREQ_CAP_REQUIRED)
    freq_all = _freq_bonus(
        resume_counts_weighted,
        (jd_required | jd_preferred) - set(req_missing) - set(pref_missing),
        FREQ_CAP_OVERALL
    )

    # Section hygiene: require core four
    needed = ["experience", "skills", "projects", "education"]
    present = sum(1 for s in needed if sections_presence.get(s, False))
    section_hygiene = round(100.0 * present / len(needed), 1)

    # Penalties
    words_count = len(_words(rt))
    length_pen  = _length_penalty(words_count)
    stuff_points = _stuffing_penalty_amount(resume_counts_weighted, words_count)

    # Final score
    score = (
        W_REQUIRED_COVERAGE * req_cov +
        W_OVERALL_COVERAGE  * overall_cov +
        W_FREQ_REQ          * freq_req +
        W_FREQ_ALL          * freq_all +
        W_SECTION_HYGIENE   * section_hygiene
    )
    if length_pen:
        score -= PENALTY_LENGTH
    if stuff_points > 0:
        score -= stuff_points
    score = max(0.0, min(100.0, round(score, 1)))

    matched_keywords = sorted(list(((jd_required | jd_preferred) - set(req_missing) - set(pref_missing)) | set(ai_matches_list)))
    missing_keywords = sorted(list(set(req_missing + pref_missing)))

    gaps_by_cat: Dict[str, List[str]] = defaultdict(list)
    for m in req_missing + pref_missing:
        cat = _CANON_CATEGORY.get(m, "other")
        gaps_by_cat[cat].append(m)

    # Simple job title heuristic
    job_title = job_description.splitlines()[0].strip() if job_description.strip() else None
    page_count_estimate = len(resume_text) / 1800 if resume_text else 0

    suggested = _suggestions(
        req_missing=req_missing,
        pref_missing=pref_missing,
        resume_text=resume_text,
        sections=sections_text,
        page_count=page_count_estimate,
        job_title=job_title,
        bullet_ratio=bullet_ratio_pct,  # percent
        verb_ratio=verb_ratio_pct,      # percent
    )

    components = {
        "required_coverage": req_cov,
        "overall_coverage": overall_cov,
        "frequency_bonus_required": freq_req,
        "frequency_bonus_overall": freq_all,
        "section_hygiene": section_hygiene,
        "bullets_ratio_%": bullet_ratio_pct,
        "action_verb_ratio_%": verb_ratio_pct,
        "length_penalty_applied": bool(length_pen),
        "stuffing_penalty_applied": stuff_points > 0,
        "ai_on": ai_on,
        "ai_semantic_matches": len(ai_matches_list),
    }

    return {
        "score": score,
        "components": components,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "required_missing": req_missing,
        "preferred_missing": pref_missing,
        "gaps_by_category": dict(gaps_by_cat),
        "suggested_bullets": suggested,
        "avg_sentence_length": round(
            sum(len(s.split()) for s in _sentences(rt) if s.strip()) /
            max(1, len([1 for s in _sentences(rt) if s.strip()]))
        , 1),
        "section_presence": sections_presence,   # canonicalized
        "top_resume_terms": _top_terms(rt, k=20),
        "jd_skill_terms": sorted(list(jd_required | jd_preferred)),
        "ai_matched_terms": ai_matches_list,
        "ai_similarities": ai_sim_map,
    }

# ----------------- Top terms -----------------
def _top_terms(text: str, k: int = 20) -> List[str]:
    w = [t for t in _words(text) if len(t) >= 2]
    cnt = Counter(w)
    return [x for x, _ in cnt.most_common(k)]
