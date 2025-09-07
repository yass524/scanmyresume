# ats_core.py
# Upgraded, dependency-light scoring engine + optional semantic (embeddings) matching.

import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

# ----------------------------
# Optional embeddings (toggle with EMB_ON=1)
# ----------------------------
try:
    from sentence_transformers import SentenceTransformer, util  # pip install sentence-transformers
    _EMB_MODEL_NAME = os.environ.get("EMB_MODEL", "all-MiniLM-L6-v2")
    _EMB_ON = os.environ.get("EMB_ON", "0") == "1"
    _emb_model = SentenceTransformer(_EMB_MODEL_NAME) if _EMB_ON else None
except Exception:
    _emb_model = None
    _EMB_ON = False

def _emb_threshold() -> float:
    try:
        return float(os.environ.get("EMB_THRESH", "0.62"))
    except Exception:
        return 0.62

# ----------------------------
# Editable taxonomy & synonyms
# ----------------------------
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "programming_languages": [
        "python","c","c++","java","javascript","typescript","matlab","r","go","ruby",
        "rust","kotlin","swift","bash","shell","powershell"
    ],
    "ml_ai": [
        "machine learning","deep learning","pytorch","tensorflow","keras","scikit-learn",
        "opencv","computer vision","nlp","transformers","xgboost","lightgbm","pca","svd",
        "yolo","yolov5","yolov8","onnx","tensorrt","openvino","hugging face","langchain",
        "rag","faiss","chroma","llamaindex","cuda","kalman filter","ukf","ekf"
    ],
    "data": [
        "sql","postgresql","mysql","mongodb","bigquery","snowflake","nosql","data analysis",
        "pandas","numpy","matplotlib","plotly","tableau","power bi","dbt","etl","data warehouse"
    ],
    "cloud_devops": [
        "aws","azure","gcp","docker","kubernetes","linux","git","github","gitlab","ci","cd",
        "jenkins","terraform","ansible","mlflow","airflow","prefect","sagemaker","vertex ai",
        "ec2","s3","lambda","api gateway","cloudwatch","cloudformation","iam","ecr","ecs","eks"
    ],
    "web": [
        "react","next.js","node","express","flask","fastapi","rest api","graphql","tailwind",
        "socket.io","jwt","oauth","nestjs","vite"
    ],
    "embedded_robotech": [
        "arduino","stm32","esp32","arm cortex","rtos","can bus","spi","i2c","uart","ros","ros2",
        "opencv","yolo","kalman filter","ukf","ekf"
    ],
    "general": [
        "algorithms","data structures","oop","design patterns","unit testing","integration testing",
        "agile","scrum","jira","communication","leadership","problem solving","systems design"
    ]
}

# Synonyms/aliases → canonical form
SYNONYMS: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "nodejs": "node",
    "node.js": "node",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "hf": "hugging face",
    "soc": "state of charge",
    "opencv-python": "opencv",
    "pgsql": "postgresql",
    "postgre": "postgresql",
    "k8s": "kubernetes",
    "g cloud": "gcp",
    "google cloud": "gcp",
    "ms azure": "azure",
    "ci/cd": "ci",
    "rest": "rest api",
    "api": "rest api",
    "cv": "computer vision",
    "comp vision": "computer vision",
    "torch": "pytorch",
    "pytorch-lightning": "pytorch",
    "amazon web services": "aws",
    "aws ec2": "ec2",
    "aws s3": "s3"
}

SECTION_HINTS = [
    "experience","education","projects","skills","certifications",
    "publications","summary","profile",
    "work experience","professional experience","skills & tools","tech stack",
    "professional summary","objective","profile summary"
]

ACTION_VERBS = {
    "built","designed","implemented","deployed","optimized","led","created","launched",
    "improved","reduced","increased","delivered","automated","migrated","scaled","integrated",
    "trained","fine-tuned","evaluated","benchmarked"
}

BULLET_PATTERN = r"^(\•|\-|\*|–|—|·|▪|●|\d+\.)\s"

# ----------------------------
# Text utilities
# ----------------------------
def normalize_text(text: str) -> str:
    text = (text or "").replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(text: str) -> List[str]:
    raw = re.findall(r"[a-z0-9\+\.\-#]+", (text or "").lower())
    return [t.rstrip(".,;:!") for t in raw if t.rstrip(".,;:!")]

def apply_synonyms(tokens: List[str]) -> List[str]:
    return [SYNONYMS.get(t, t) for t in tokens]

def expand_phrases(text: str) -> Set[str]:
    toks = apply_synonyms(tokenize(text))
    unigrams = set(toks)
    bigrams  = set(" ".join(toks[i:i+2]) for i in range(len(toks)-1))
    trigrams = set(" ".join(toks[i:i+3]) for i in range(len(toks)-2))
    return unigrams | bigrams | trigrams

def taxonomy_terms() -> Set[str]:
    terms: Set[str] = set()
    for arr in SKILL_TAXONOMY.values():
        for term in arr:
            terms.add(term.lower())
    return terms

# ----------------------------
# JD parsing (required vs preferred)
# ----------------------------
REQUIRED_PATTERNS = [
    r"\bmust have\b", r"\brequired\b", r"\bwe require\b", r"\bminimum\b",
    r"\bat least\b", r"\bstrong experience in\b", r"\bexperience with\b"
]
PREFERRED_PATTERNS = [
    r"\bnice to have\b", r"\bpreferred\b", r"\bbonus\b", r"\bplus\b", r"\bgood to have\b"
]

def _lines(text: str) -> List[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]

def _pick_lines_by_patterns(text: str, patterns: List[str]) -> List[str]:
    lines = _lines(text.lower())
    return [ln for ln in lines if any(re.search(p, ln) for p in patterns)]

def extract_jd_terms(jd_text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    n_jd = normalize_text(jd_text)
    phrases = expand_phrases(n_jd)
    tax = taxonomy_terms()

    jd_terms_all = set(p for p in phrases if p in tax)

    req_lines = _pick_lines_by_patterns(jd_text, REQUIRED_PATTERNS)
    pref_lines = _pick_lines_by_patterns(jd_text, PREFERRED_PATTERNS)

    req_terms = set(p for ln in req_lines for p in expand_phrases(ln) if p in tax)
    pref_terms = set(p for ln in pref_lines for p in expand_phrases(ln) if p in tax)

    req_terms &= jd_terms_all
    pref_terms &= jd_terms_all

    return jd_terms_all, req_terms, pref_terms

# ----------------------------
# Matching & frequency
# ----------------------------
def extract_resume_terms(resume_text: str) -> Set[str]:
    return set(p for p in expand_phrases(normalize_text(resume_text)) if p in taxonomy_terms())

def word_freq(text: str) -> Counter:
    return Counter(apply_synonyms(tokenize(text)))

def semantic_hits(jd_terms: Set[str], resume_text: str) -> Set[str]:
    """Return JD terms considered 'matched' by semantic similarity."""
    if not (_EMB_ON and _emb_model and jd_terms):
        return set()
    cand = list(expand_phrases(resume_text))  # expand uses normalize internally
    cand = [c for c in cand if 2 <= len(c) <= 40]  # filter noise
    if not cand:
        return set()

    jd_list = list(jd_terms)
    emb_jd = _emb_model.encode(jd_list, normalize_embeddings=True, convert_to_tensor=True)
    emb_c  = _emb_model.encode(cand,    normalize_embeddings=True, convert_to_tensor=True)
    sim = util.cos_sim(emb_jd, emb_c)
    thresh = _emb_threshold()

    matched = set()
    for i, term in enumerate(jd_list):
        j = int(sim[i].argmax())
        if float(sim[i][j]) >= thresh:
            matched.add(term)
    return matched

# ----------------------------
# Hygiene & heuristics
# ----------------------------
def section_signals(text: str) -> Dict[str, bool]:
    lower = (text or "").lower()
    return {s: (s in lower) for s in SECTION_HINTS}

def avg_sentence_len(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[\.!?]", text or "") if s.strip()]
    return (sum(len(s.split()) for s in sentences)/len(sentences)) if sentences else 0.0

def bullet_share(text: str) -> float:
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if not lines: return 0.0
    bullets = sum(1 for l in lines if re.match(BULLET_PATTERN, l.strip()))
    return bullets / max(1, len(lines))

def action_verb_ratio(text: str) -> float:
    lines = _lines(text)
    if not lines: return 0.0
    starts = 0
    for l in lines:
        tok = (tokenize(l) or [""])[0]
        if tok in ACTION_VERBS:
            starts += 1
    return starts / max(1, len(lines))

def stuffing_risk(freq: Counter) -> bool:
    if not freq: return False
    total = sum(freq.values())
    top = freq.most_common(1)[0][1]
    return total >= 30 and (top / total) > 0.22

# ----------------------------
# Main scoring
# ----------------------------
def compute_score(resume_text: str, jd_text: str) -> Dict:
    resume_n = normalize_text(resume_text)
    jd_n     = normalize_text(jd_text)

    jd_all, jd_required, jd_preferred = extract_jd_terms(jd_text)

    # Exact/phrase matches
    res_terms_exact = extract_resume_terms(resume_text)

    # Semantic near-matches (optional)
    sem_added = semantic_hits(jd_all, resume_text)
    res_terms = set(res_terms_exact) | set(sem_added)
    ai_extra_count = len(res_terms - res_terms_exact)

    # Coverage
    def cov(have: Set[str], need: Set[str]) -> float:
        return len(have & need) / len(need) if need else 1.0

    req_cov = cov(res_terms, jd_required)
    all_cov = cov(res_terms, jd_all)

    # Frequency bonus
    wf = word_freq(resume_n)
    def freq_bonus_for(terms: Set[str]) -> float:
        if not terms: return 1.0
        raw = 0
        for t in terms:
            tokens = tokenize(t)
            raw += min(sum(wf.get(tok, 0) for tok in tokens), 3)
        denom = 3 * max(len(terms), 1)
        return raw / denom

    freq_req = freq_bonus_for(jd_required)
    freq_all = freq_bonus_for(jd_all)

    # Hygiene
    sections = section_signals(resume_text)
    section_score = sum(1 for v in sections.values() if v) / len(sections) if sections else 1.0
    avg_len = avg_sentence_len(resume_n)
    bullet_fraction = bullet_share(resume_text)
    action_ratio = action_verb_ratio(resume_text)

    length_penalty = 0.07 if avg_len > 28 else 0.0
    bullet_nudge   = 0.03 if bullet_fraction >= 0.25 else 0.0
    action_nudge   = 0.03 if action_ratio >= 0.15 else 0.0
    stuffing_flag  = stuffing_risk(wf)
    stuffing_pen   = 0.05 if stuffing_flag else 0.0

    score = (
        req_cov * 0.45 +
        all_cov * 0.25 +
        ((freq_req + freq_all) / 2.0) * 0.10 +
        section_score * 0.16 +    # slightly higher weight for structure
        bullet_nudge + action_nudge
    ) * 100
    score = max(0, score - (length_penalty * 100) - (stuffing_pen * 100))
    score = round(score, 1)

    matched = sorted(list(res_terms & jd_all))
    missing_all = sorted(list(jd_all - res_terms))
    missing_required = sorted(list(jd_required - res_terms))
    missing_preferred = sorted(list((jd_preferred - res_terms) - set(missing_required)))

    cat_by_term = {}
    for cat, terms in SKILL_TAXONOMY.items():
        for t in terms:
            cat_by_term[t] = cat
    gaps_by_cat = defaultdict(list)
    for t in missing_all:
        gaps_by_cat[cat_by_term.get(t, "other")].append(t)

    suggestions: List[str] = []
    if missing_required:
        for kw in missing_required[:3]:
            suggestions.append(
                f"Add a tangible example with '{kw}' (e.g., 'Implemented {kw} to achieve measurable improvement')."
            )
    if missing_preferred:
        for kw in missing_preferred[:2]:
            suggestions.append(
                f"Add brief exposure to '{kw}' (mini-deploy, lab, or course) if relevant to the role."
            )
    if avg_len > 28:
        suggestions.append(f"Shorten sentences (avg ~{avg_len:.1f} words). Aim for 12–24 word bullets.")
    if bullet_fraction < 0.25:
        suggestions.append("Use bullets (•/–) for experience; target ≥25% of lines as bullets.")
    if action_ratio < 0.15:
        suggestions.append("Start bullets with action verbs (Built, Implemented, Optimized, Led...).")
    if stuffing_flag:
        suggestions.append("Reduce repeated keyword spam; show outcomes instead of repeating the same tech.")

    stop = {"and","or","the","a","an","to","of","in","on","for","with","by","i","we","they","you","from","at"}
    top_terms = [w for w, c in wf.most_common(120) if w not in stop][:20]

    components = {
        "required_coverage": round(req_cov * 100, 1),
        "overall_coverage": round(all_cov * 100, 1),
        "frequency_bonus_required": round(freq_req * 100, 1),
        "frequency_bonus_overall": round(freq_all * 100, 1),
        "section_hygiene": round(section_score * 100, 1),
        "bullets_ratio_%": round(bullet_fraction * 100, 1),
        "action_verb_ratio_%": round(action_ratio * 100, 1),
        "length_penalty_applied": avg_len > 28,
        "stuffing_penalty_applied": stuffing_flag,
        "ai_on": _EMB_ON,
        "ai_semantic_matches": int(ai_extra_count)
    }

    return {
        "score": score,
        "components": components,
        "matched_keywords": matched,
        "missing_keywords": missing_all,
        "required_missing": missing_required,
        "preferred_missing": missing_preferred,
        "gaps_by_category": dict(gaps_by_cat),
        "suggested_bullets": suggestions[:5],
        "avg_sentence_length": round(avg_len, 1),
        "section_presence": sections,
        "top_resume_terms": top_terms,
        "jd_skill_terms": sorted(list(jd_all))
    }
SYNONYMS.update({
    "image processing": "computer vision",
    "vision pipeline": "computer vision",

    "containers": "docker",
    "containerized": "docker",
    "containerization": "docker",
})
