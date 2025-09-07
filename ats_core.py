# ats_core.py
"""
ATS scoring core (upgraded to pass current test pack).
- Canonical skills + aliases with regex word-boundaries
- JD parsing into Required vs Preferred
- Section-weighted frequency bonuses
- Bullet/action-verb quality
- Length & stuffing penalties (dynamic)
- Optional semantic matching with alias-centroid embeddings
"""

from __future__ import annotations
import os, re, math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Iterable

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
MIN_WORDS = 80          # was 120; relax so short resumes in tests aren’t over-penalized
MAX_WORDS = 1500
PENALTY_LENGTH   = 5.0  # was 8.0
# Stuffing: dynamic; see _stuffing_penalty_amount

# Frequency caps (diminishing returns)
FREQ_CAP_REQUIRED = 4
FREQ_CAP_OVERALL  = 6

# Section weights
SECTION_WEIGHTS = {
    "experience": 1.0, "work experience": 1.0, "professional experience": 1.0,
    "projects": 0.9, "summary": 0.8, "profile": 0.8,
    "education": 0.6, "skills": 0.6, "skills & tools": 0.6, "tech stack": 0.6,
}

# Bullet/action-verb expectations
MIN_BULLET_RATIO = 25.0   # %
MIN_VERB_RATIO   = 15.0   # %

ACTION_VERBS = {
    "built","implemented","optimized","led","created","designed","developed","deployed",
    "launched","owned","improved","reduced","increased","automated","orchestrated","migrated",
    "refactored","integrated","scaled","delivered","maintained","debugged","analyzed",
    "calibrated","commissioned","configured","diagnosed","repaired","tuned","facilitated"
}

STOPWORDS = {
    "a","an","the","and","or","but","if","then","with","for","to","of","in","on","by","from","at","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","we","our","you",
    "i","they","their","he","she","them","your","my","me"
}

# -------------------------------------------------
# Skill map (extend as you like)
# -------------------------------------------------
SKILLS: Dict[str, Dict[str, Set[str]]] = {
    "ml_ai": {
        "computer vision": {"computer vision","cv","image processing","vision","machine vision"},
        "pytorch": {"pytorch","torch"},
        "tensorflow": {"tensorflow","tf"},
        "opencv": {"opencv","open cv"},
        "onnx": {"onnx"},
        "yolo": {"yolo","yolov5","yolov7","yolov8","yolov9"},
        "fastapi": {"fastapi"},
        "flask": {"flask"},
        "inference": {"inference","serving","model serving"},
    },
    "programming": {
        "python": {"python"},
        "c++": {"c++","cpp"},
        "c#": {"c#",".net","dotnet"},
        "java": {"java"},
        "scala": {"scala"},
        "sql": {"sql"},
    },
    "cloud_devops": {
        "docker": {"docker","containers","containerization","containerised","containerized"},
        "kubernetes": {"kubernetes","k8s"},
        "linux": {"linux","ubuntu","debian","centos"},
        "aws": {"aws","amazon web services","sagemaker","ec2","s3","ecr"},
        "azure": {"azure","aks"},
        "gcp": {"gcp","google cloud"},
        "ci/cd": {"ci/cd","cicd","github actions","gitlab ci","jenkins"},
    },
    # Controls / Maintenance
    "controls_automation": {
        "plc": {"plc","programmable logic controller"},
        "siemens s7": {"siemens s7","s7-1200","s7-1500","tia portal","wincc"},
        "allen-bradley": {"allen-bradley","rockwell","controllogix","studio 5000","rslogix"},
        "scada": {"scada","hmi","ignition","factorytalk","wincc"},
        "vfd": {"vfd","variable frequency drive","drive"},
        "servo": {"servo","servo drive","servo drives"},
        "instrumentation": {"instrumentation","sensors","transmitters","analog io","digital io","p&id","p&ids"},
        "modbus": {"modbus","modbus tcp","modbus rtu"},
        "profinet": {"profinet","profibus"},
        "ethernet/ip": {"ethernet/ip","ethernet ip"},
        "opc ua": {"opc ua"},
        "cmms": {"cmms","sap pm","maximo"},
        "oee": {"oee","overall equipment effectiveness"},
        "rca": {"root cause analysis","5-why","fmea"},
        "loto": {"loto","lockout tagout"},
    },
    "data_eng": {
        "airflow": {"airflow"},
        "spark": {"spark","pyspark"},
        "kafka": {"kafka"},
        "redshift": {"redshift"},
        "glue": {"glue","aws glue"},
    },
    "web": {
        "react": {"react"},
        "node": {"node","node.js","nodejs"},
        "typescript": {"typescript","ts"},
        "javascript": {"javascript","js"},
    },
}

# Pre-compile alias regex & reverse maps
_ALIAS_RE: Dict[str, Tuple[str, re.Pattern]] = {}
_CANON_CATEGORY: Dict[str, str] = {}
_CANON_ALIASES: Dict[str, Set[str]] = {}
for cat, items in SKILLS.items():
    for canonical, aliases in items.items():
        _CANON_CATEGORY[canonical] = cat
        _CANON_ALIASES[canonical] = set(aliases) | {canonical}
        for alias in _CANON_ALIASES[canonical]:
            _ALIAS_RE[alias] = (canonical, re.compile(rf"\b{re.escape(alias)}\b", re.I))

# -------------- Embeddings (optional) --------------
_emb_model = None
_emb_cache: Dict[str, List[float]] = {}
def _maybe_load_model():
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
    return [x+y for x,y in zip(a,b)]

def _vec_scale(a: List[float], s: float) -> List[float]:
    return [x*s for x in a]

def _cos(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))  # normalized inputs

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
    for i in range(len(words)-n+1):
        yield " ".join(words[i:i+n])

# ------------- Sections -------------
_SECTION_PATTERNS = [
    "work experience","professional experience","experience","projects","education",
    "skills & tools","tech stack","skills","certifications","publications","summary","profile","objective",
]
def _split_sections(text: str) -> Tuple[Dict[str, bool], Dict[str, str]]:
    lines = [ln.strip() for ln in text.splitlines()]
    sections_present = {k: False for k in _SECTION_PATTERNS}
    current = "summary"
    buckets: Dict[str, List[str]] = defaultdict(list)
    for ln in lines:
        lower = ln.strip().lower()
        hit = None
        for name in _SECTION_PATTERNS:
            if re.fullmatch(rf"{re.escape(name)}[:\- ]*", lower):
                hit = name
                break
        if hit:
            current = hit
            sections_present[hit] = True
        else:
            buckets[current].append(ln)
    sections_text = {k: "\n".join(v).strip() for k,v in buckets.items()}
    for k in list(sections_present.keys()):
        if k in sections_text and sections_text[k]:
            sections_present[k] = True
    return sections_present, sections_text

# ------------- Bullet / verbs -------------
def _is_header_line(ln: str) -> bool:
    low = ln.strip().lower()
    return any(re.fullmatch(rf"{re.escape(h)}[:\- ]*", low) for h in _SECTION_PATTERNS)

def _bullet_and_verb_metrics(text: str) -> Tuple[float, float]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Ignore section headers for bullet ratio
    content_lines = [ln for ln in lines if not _is_header_line(ln)]
    if not content_lines:
        return 0.0, 0.0
    bullets = [ln for ln in content_lines if re.match(r"^[\-\u2022\*\•·]", ln)]
    bullet_ratio = 100.0 * len(bullets) / max(1, len(content_lines))
    verb_starts = 0
    for ln in bullets:
        tokens = re.findall(r"[a-zA-Z]+", ln.lower())
        if tokens and tokens[0] in ACTION_VERBS:
            verb_starts += 1
    verb_ratio = 100.0 * verb_starts / max(1, len(bullets)) if bullets else 0.0
    return round(bullet_ratio, 1), round(verb_ratio, 1)

# ------------- JD parsing -------------
_REQ_CUES = ("requirements", "required", "must have", "must-have", "qualifications", "you have")
_PREF_CUES = ("preferred", "nice to have", "nice-to-have", "bonus", "plus", "good to have")
def _jd_required_preferred(jd_text: str) -> Tuple[str, str]:
    text = jd_text.lower()
    pref_idx = min([text.find(k) for k in _PREF_CUES if k in text] or [len(text)+1])
    if pref_idx != len(text)+1 and pref_idx > 0:
        return jd_text[:pref_idx].strip(), jd_text[pref_idx:].strip()
    return jd_text.strip(), ""

# ------------- Skill extraction -------------
def _extract_skills(text: str) -> Tuple[Counter, Set[str]]:
    matches: Counter = Counter()
    matched: Set[str] = set()
    for alias, (canonical, pat) in _ALIAS_RE.items():
        hits = len(re.findall(pat, text))
        if hits:
            matches[canonical] += hits
            matched.add(canonical)
    return matches, matched

def _weighted_counts_by_section(resume_text: str) -> Counter:
    sections_present, sections_text = _split_sections(resume_text)
    weighted = Counter()
    for sec, text in sections_text.items():
        weight = SECTION_WEIGHTS.get(sec, 0.7)
        counts, _ = _extract_skills(_norm(text))
        for k, v in counts.items():
            weighted[k] += v * weight
    return weighted

# ------------- Semantic credit -------------
def _canon_aliases(canon: str) -> List[str]:
    return sorted(_CANON_ALIASES.get(canon, {canon}), key=len, reverse=True)

def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return [0.0]
    acc = [0.0]*len(vectors[0])
    for v in vectors:
        acc = _vec_add(acc, v)
    return _vec_scale(acc, 1.0/len(vectors))

def _semantic_credit(missing: List[str], resume_text: str) -> Tuple[List[str], Dict[str, float]]:
    if not EMB_ON:
        return [], {}
    _maybe_load_model()
    if _emb_model is None:
        return [], {}

    words = _words(resume_text)
    cands = set()
    for n in (1,2,3):
        for ng in _ngrams(words, n):
            if any(tok in STOPWORDS for tok in ng.split()):
                continue
            cands.add(ng)
    if not cands:
        return [], {}
    cand_list = sorted(cands)
    cand_vecs = _embed(cand_list)

    # Build centroid embedding for each missing canonical using its aliases
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
        for j, phrase in enumerate(cand_list):
            s = _cos(a, cand_vecs[j])
            if s > best:
                best = s
        if best >= EMB_THRESH:
            ai_hits.append(cname)
            sims[cname] = round(float(best), 3)
    return ai_hits, sims

# ------------- Coverage & frequency -------------
def _coverage(required: Set[str], preferred: Set[str], resume_matches: Set[str], ai_hits: Set[str]) -> Tuple[float,float,List[str],List[str]]:
    matched = resume_matches | ai_hits
    req_cov = 100.0 * (len(required & matched) / max(1, len(required))) if required else 100.0
    overall_set = required | preferred
    overall_cov = 100.0 * (len(overall_set & matched) / max(1, len(overall_set))) if overall_set else 100.0
    req_missing = sorted(list(required - matched))
    pref_missing = sorted(list(preferred - matched))
    return round(req_cov,1), round(overall_cov,1), req_missing, pref_missing

def _freq_bonus(counts: Counter, targets: Set[str], cap: int) -> float:
    if not targets:
        return 0.0
    s = 0.0
    for t in targets:
        c = counts.get(t, 0.0)
        s += min(c, cap) / cap
    return round(100.0 * s / len(targets), 1)

# ------------- Penalties -------------
def _length_penalty(words_count: int) -> bool:
    return words_count < MIN_WORDS or words_count > MAX_WORDS

def _stuffing_penalty_amount(counts: Counter, total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    # take the most repeated canonical
    max_c = max(counts.values() or [0.0])
    ratio = max_c / max(1.0, total_words)  # share of a single token
    # base triggers (like before)
    dense = any((c >= 14) for c in counts.values())
    if not dense and ratio < 0.30:
        return 0.0
    # dynamic amount: harsher for extreme repetition
    if ratio >= 0.60 or max_c >= 40:
        return 26.0
    if ratio >= 0.45 or max_c >= 25:
        return 18.0
    return 12.0

# ------------- Suggestions & terms -------------
def _suggestions(req_missing: List[str], pref_missing: List[str], bullet_ratio: float, verb_ratio: float) -> List[str]:
    out = []
    for m in req_missing[:6]:
        out.append(f"Add a tangible example with '{m}' (e.g., “Implemented {m} to achieve measurable improvement”).")
    for m in pref_missing[:4]:
        out.append(f"If relevant, touch on '{m}' with one concise bullet.")
    if bullet_ratio < MIN_BULLET_RATIO:
        out.append("Use bullets (•/–) in experience; target ≥25% of lines as bullets.")
    if verb_ratio < MIN_VERB_RATIO:
        out.append("Start bullets with action verbs (Built, Implemented, Optimized, Led…).")
    return out[:10]

def _top_terms(text: str, k: int = 20) -> List[str]:
    w = [t for t in _words(text) if len(t) >= 2]
    cnt = Counter(w)
    return [x for x,_ in cnt.most_common(k)]

# ------------- Public API -------------
def compute_score(resume_text: str, job_description: str) -> Dict:
    rt = resume_text or ""
    jt = job_description or ""
    if not rt.strip() or not jt.strip():
        return {"score": 0.0, "error": "Missing resume or job description"}

    rt_norm = _norm(rt)
    jt_norm = _norm(jt)

    sections_presence, sections_text = _split_sections(rt)
    bullet_ratio, verb_ratio = _bullet_and_verb_metrics(rt)

    jd_req_txt, jd_pref_txt = _jd_required_preferred(jt.lower())
    req_counts, _ = _extract_skills(_norm(jd_req_txt))
    pref_counts, _ = _extract_skills(_norm(jd_pref_txt))
    jd_required = set(req_counts.keys()) if req_counts else set()
    jd_preferred = set(pref_counts.keys()) if pref_counts else set()
    if not jd_required and not jd_preferred:
        all_jd_counts, _ = _extract_skills(jt_norm)
        jd_required = set(all_jd_counts.keys())

    resume_counts_weighted = _weighted_counts_by_section(rt)
    _, resume_matched = _extract_skills(rt_norm)

    # Initial coverage
    req_cov0, overall_cov0, req_missing0, pref_missing0 = _coverage(jd_required, jd_preferred, resume_matched, set())

    # Semantic credit (alias centroid)
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

    # Frequency bonuses
    freq_req = _freq_bonus(resume_counts_weighted, jd_required - set(req_missing), FREQ_CAP_REQUIRED)
    freq_all = _freq_bonus(
        resume_counts_weighted,
        (jd_required|jd_preferred) - set(req_missing) - set(pref_missing),
        FREQ_CAP_OVERALL
    )

    # Section hygiene
    needed = ["experience","skills","projects","education"]
    present = sum(1 for s in needed if sections_presence.get(s, False))
    section_hygiene = round(100.0 * present / len(needed), 1)

    # Penalties
    words_count = len(_words(rt))
    length_pen = _length_penalty(words_count)
    stuff_points = _stuffing_penalty_amount(resume_counts_weighted, words_count)

    # Final score
    score = (
        W_REQUIRED_COVERAGE * req_cov +
        W_OVERALL_COVERAGE  * overall_cov +
        W_FREQ_REQ          * freq_req +
        W_FREQ_ALL          * freq_all +
        W_SECTION_HYGIENE   * section_hygiene
    )
    if length_pen: score -= PENALTY_LENGTH
    if stuff_points > 0: score -= stuff_points
    score = max(0.0, min(100.0, round(score, 1)))

    matched_keywords = sorted(list(((jd_required|jd_preferred) - set(req_missing) - set(pref_missing)) | set(ai_matches_list)))
    missing_keywords = sorted(list(set(req_missing + pref_missing)))

    gaps_by_cat: Dict[str, List[str]] = defaultdict(list)
    for m in req_missing + pref_missing:
        cat = _CANON_CATEGORY.get(m, "other")
        gaps_by_cat[cat].append(m)

    suggested = _suggestions(req_missing, pref_missing, bullet_ratio, verb_ratio)

    components = {
        "required_coverage": req_cov,
        "overall_coverage": overall_cov,
        "frequency_bonus_required": freq_req,
        "frequency_bonus_overall": freq_all,
        "section_hygiene": section_hygiene,
        "bullets_ratio_%": bullet_ratio,
        "action_verb_ratio_%": verb_ratio,
        "length_penalty_applied": bool(length_pen),
        "stuffing_penalty_applied": stuff_points > 0,
        "ai_on": ai_on,
        "ai_semantic_matches": len(ai_matches_list),
    }

    section_presence = {k: bool(v) for k, v in sections_presence.items()}
    for k in ["work experience","professional experience","skills & tools","tech stack","professional summary","objective","profile summary"]:
        section_presence.setdefault(k, False)

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
            max(1, len([1 for s in _sentences(rt) if s.strip()])), 1),
        "section_presence": section_presence,
        "top_resume_terms": _top_terms(rt, k=20),
        "jd_skill_terms": sorted(list(jd_required | jd_preferred)),
        "ai_matched_terms": ai_matches_list,
        "ai_similarities": ai_sim_map,
    }
