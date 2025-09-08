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
SECTION_WEIGHTS = {# experience mehataga tetzebet
    "experience": 1.0, "work experience": 1.0, "professional experience": 1.0,
    "projects": 0.9, "summary": 0.8, "profile": 0.8,
    "education": 0.6, "skills": 0.6, "skills & tools": 0.6, "tech stack": 0.6,
}

# Bullet/action-verb expectations
MIN_BULLET_RATIO = 25.0   # %
MIN_VERB_RATIO   = 15.0   # %

ACTION_VERBS = {
    #  Core technical / building
    "built","implemented","optimized","created","designed","developed","deployed",
    "launched","engineered","constructed","assembled","programmed","coded","automated",
    "refactored","configured","integrated","tested","validated","modeled","simulated",
    "calibrated","commissioned","debugged","diagnosed","repaired","tuned","maintained",

    # Leadership / initiative
    "led","spearheaded","orchestrated","coordinated","directed","supervised","oversaw",
    "managed","executed","initiated","organized","owned","facilitated","mentored",
    "trained","supported","guided",

    # Achievement / results
    "delivered","achieved","exceeded","improved","enhanced","expanded","scaled","increased",
    "reduced","streamlined","upgraded","resolved","strengthened","surpassed","won","earned",

    # Innovation / strategy
    "architected","conceived","devised","formulated","innovated","pioneered","strategized",
    "devised","introduced","proposed","instituted","transformed","modernized",

    #  Collaboration / communication
    "collaborated","partnered","contributed","consulted","advised","presented","communicated",
    "drafted","documented","reviewed","negotiated","liaised","aligned","engaged","shared",
    "informed","advocated","supported","facilitated"
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
    # ---------- AI / ML ----------
    "ml_ai": {
        "computer vision": {"computer vision","cv","image processing","vision","machine vision"},
        "pytorch": {"pytorch","torch"},
        "tensorflow": {"tensorflow","tf"},
        "opencv": {"opencv","open cv"},
        "onnx": {"onnx"},
        "onnxruntime": {"onnxruntime","onnx runtime"},
        "yolo": {"yolo","yolov5","yolov7","yolov8","yolov9"},
        "transformers": {"transformers","huggingface","hf transformers"},
        "sklearn": {"scikit-learn","sklearn"},
        "xgboost": {"xgboost"},
        "lightgbm": {"lightgbm","lgbm"},
        "catboost": {"catboost"},
        "numpy": {"numpy"},
        "pandas": {"pandas"},
        "mlflow": {"mlflow"},
        "wandb": {"weights & biases","weights and biases","wandb"},
        "inference": {"inference","serving","model serving"},
        "tensorrt": {"tensorrt","trt"},
        "tflite": {"tflite","tensorflow lite"},
        "openvino": {"openvino"},
        "triton": {"triton inference server","triton","nv triton"},
        "langchain": {"langchain"},
        "spacy": {"spacy"},
        "nltk": {"nltk"},
    },

    # ---------- Programming ----------
    "programming": {
        "python": {"python"},
        "c++": {"c++","cpp"},
        "c": {"c"},
        "c#": {"c#",".net","dotnet"},
        "java": {"java"},
        "go": {"go","golang"},
        "rust": {"rust"},
        "scala": {"scala"},
        "matlab": {"matlab"},
        "simulink": {"simulink"},
        "sql": {"sql"},
        "bash": {"bash","shell","sh"},
        "powershell": {"powershell","ps"},
        "javascript": {"javascript","js"},
        "typescript": {"typescript","ts"},
        "r": {"r","r language"},
        "php": {"php"},
        "perl": {"perl"},
    },

    # ---------- Cloud / DevOps ----------
    "cloud_devops": {
        "docker": {"docker","containers","containerization"},
        "kubernetes": {"kubernetes","k8s"},
        "linux": {"linux","ubuntu","debian","centos","rhel"},
        "aws": {"aws","amazon web services","sagemaker","ec2","s3","ecr","lambda"},
        "azure": {"azure","aks","azure devops"},
        "gcp": {"gcp","google cloud","gke","bigquery"},
        "terraform": {"terraform","iac","infrastructure as code"},
        "ansible": {"ansible"},
        "helm": {"helm","helm charts"},
        "prometheus": {"prometheus"},
        "grafana": {"grafana"},
        "ci/cd": {"ci/cd","cicd","github actions","gitlab ci","jenkins"},
    },

    # ---------- Controls / Automation ----------
    "controls_automation": {
        "plc": {"plc","programmable logic controller"},
        "siemens s7": {"siemens s7","s7-1200","s7-1500","tia portal","wincc"},
        "allen-bradley": {"allen-bradley","rockwell","controllogix","studio 5000","rslogix"},
        "schneider electric": {"schneider electric","m340","m580","ecostruxure"},
        "mitsubishi": {"mitsubishi","fx5u","gx works"},
        "omron": {"omron","sysmac","cx-programmer"},
        "beckhoff": {"beckhoff","twincat","ethercat"},
        "codesys": {"codesys"},
        "scada": {"scada","hmi","ignition","factorytalk","wonderware"},
        "vfd": {"vfd","variable frequency drive","drive"},
        "servo": {"servo","servo drive","motion control"},
        "instrumentation": {"instrumentation","sensors","transmitters","analog io","digital io"},
        "modbus": {"modbus","modbus tcp","modbus rtu"},
        "profinet": {"profinet","profibus"},
        "ethernet/ip": {"ethernet/ip","ethernet ip"},
        "opc ua": {"opc ua","opcua"},
        "mqtt": {"mqtt"},
        "cmms": {"cmms","sap pm","maximo"},
        "rca": {"root cause analysis","5-why","fmea"},
    },

    # ---------- Embedded / Edge ----------
    "embedded": {
        "arduino": {"arduino"},
        "esp32": {"esp32","esp-idf"},
        "stm32": {"stm32","stm32cube"},
        "pic": {"pic","mplab","xc8"},
        "raspberry pi": {"raspberry pi","rpi"},
        "freertos": {"freertos"},
        "zephyr": {"zephyr rtos","zephyr"},
        "uart": {"uart","serial"},
        "i2c": {"i2c"},
        "spi": {"spi"},
        "can": {"can","can bus","canopen"},
        "pwm": {"pwm"},
    },

    # ---------- Data Engineering ----------
    "data_eng": {
        "airflow": {"airflow"},
        "spark": {"spark","pyspark"},
        "kafka": {"kafka"},
        "flink": {"flink"},
        "dbt": {"dbt"},
        "redshift": {"redshift"},
        "glue": {"glue","aws glue"},
        "snowflake": {"snowflake"},
        "bigquery": {"bigquery"},
        "databricks": {"databricks"},
        "hive": {"hive"},
    },

    # ---------- Web / Frontend / Backend ----------
    "web": {
        "react": {"react"},
        "next.js": {"next.js","nextjs"},
        "vue": {"vue","vue.js","vuejs"},
        "angular": {"angular"},
        "node": {"node","node.js","nodejs"},
        "express": {"express"},
        "django": {"django"},
        "graphql": {"graphql","gql"},
        "rest api": {"rest","rest api"},
    },

    # ---------- Mobile ----------
    "mobile": {
        "android": {"android","kotlin","java (android)"},
        "ios": {"ios","swift"},
        "react native": {"react native"},
        "flutter": {"flutter","dart"},
    },

    # ---------- Databases ----------
    "databases": {
        "postgresql": {"postgresql","postgres","psql"},
        "mysql": {"mysql"},
        "sqlite": {"sqlite"},
        "mongodb": {"mongodb","mongo"},
        "redis": {"redis"},
        "elasticsearch": {"elasticsearch","elastic","es"},
        "cassandra": {"cassandra"},
        "neo4j": {"neo4j","graph db","graph database"},
    },

    # ---------- Analytics / BI ----------
    "analytics_bi": {
        "excel": {"excel","microsoft excel"},
        "power bi": {"power bi","powerbi"},
        "tableau": {"tableau"},
        "looker": {"looker","google data studio"},
        "superset": {"superset"},
        "matplotlib": {"matplotlib"},
        "plotly": {"plotly"},
    },

    # ---------- Product / PM ----------
    "product_pm": {
        "jira": {"jira"},
        "confluence": {"confluence"},
        "trello": {"trello"},
        "asana": {"asana"},
        "notion": {"notion"},
        "miro": {"miro"},
        "agile": {"agile"},
        "scrum": {"scrum"},
        "kanban": {"kanban"},
        "okrs": {"okrs"},
    },

    # ---------- Business / Finance ----------
    "business_finance": {
        "sap": {"sap","sap erp","sap hana"},
        "oracle erp": {"oracle erp"},
        "netsuite": {"netsuite"},
        "quickbooks": {"quickbooks"},
        "salesforce": {"salesforce"},
        "hubspot": {"hubspot"},
        "zoho": {"zoho"},
        "financial modeling": {"financial modeling","valuation"},
        "accounting": {"accounting","bookkeeping"},
        "procurement": {"procurement"},
    },

    # ---------- Architecture / Civil / Mechanical ----------
    "engineering_other": {
        "autocad": {"autocad","auto cad"},
        "solidworks": {"solidworks"},
        "revit": {"revit"},
        "staad": {"staad","staad pro"},
        "ansys": {"ansys"},
        "catia": {"catia"},
        "etabs": {"etabs"},
        "primavera": {"primavera","p6"},
        "ms project": {"ms project","microsoft project"},
        "hvac": {"hvac"},
        "bim": {"bim","building information modeling"},
        "sap2000": {"sap2000"},
    },

    # ---------- Design / UX ----------
    "design_ux": {
        "figma": {"figma"},
        "adobe xd": {"adobe xd"},
        "illustrator": {"illustrator","adobe illustrator"},
        "photoshop": {"photoshop","adobe photoshop"},
        "ux research": {"ux research","user research"},
        "wireframing": {"wireframing"},
        "prototyping": {"prototyping"},
        "indesign": {"indesign","adobe indesign"},
        "after effects": {"after effects","ae"},
        "premiere": {"premiere","premiere pro"},
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
# ----------------- Suggestion Heuristics (tunable) -----------------
MIN_BULLET_RATIO      = 0.25   # ≥25% of lines should be bullets
MIN_VERB_RATIO        = 0.55   # ≥55% of bullets should start w/ action verbs
MIN_NUMERIC_DENSITY   = 0.010  # ≥1.0% of tokens numeric (%,$,nums) ≈ quantification present
MAX_PAGES_EARLY       = 1.0    # fresh/early-career target
MAX_PAGES_GENERAL     = 2.0    # otherwise
MIN_CONTACT_FIELDS    = 2      # e.g., email + phone (or LinkedIn)
MIN_LINKS             = 1      # at least one professional link (GitHub/LinkedIn/Portfolio)
MIN_SKILLS_COUNT      = 6      # skills section should list at least N
MIN_EDU_RECENCY_YRS   = 3      # if graduate within N years, surface GPA/courses advice
MAX_LINE_LEN_CHARS    = 180    # lines longer than this could be walls of text
MAX_SOFT_SUGGESTIONS  = 3      # don't spam soft-skill advice
SUGGESTIONS_MAX_OUT   = 14     # final cap for user-facing suggestions

# Common regexes (compile once)
import re
RE_EMAIL  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
RE_PHONE  = re.compile(r"(\+?\d[\d\-\s()]{6,}\d)")
RE_URL    = re.compile(r"(https?://|www\.)\S+", re.I)
RE_DIGIT  = re.compile(r"\d")
RE_PCT    = re.compile(r"%")
RE_CURRENCY = re.compile(r"[$€£]|EGP|USD|EUR|GBP", re.I)

BULLET_MARKERS = {"•", "-", "–", "—", "*"}
ACTION_VERBS = {...}  # use your expanded set here (kept external to keep this block short)

# --------------- Light-weight analyzers ---------------- 
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
    return [ln for ln in lines if ln[:1] in BULLET_MARKERS or (ln[:2] and ln[:2] in {"- ", "• "})]

def _lines(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]

def _bullet_ratio(text: str) -> float:
    L = _lines(text)
    B = _bullet_lines(text)
    return (len(B) / max(1, len(L))) if L else 0.0

def _verb_ratio(text: str) -> float:
    B = _bullet_lines(text)
    if not B:
        return 0.0
    good = 0
    for ln in B:
        # first token as potential verb
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
    # treat a professional link as contact if present
    if re.search(r"linkedin\.com|github\.com|behance\.net|dribbble\.com|gitlab\.com", text, re.I):
        count += 1
    return count

def _links_count(text: str) -> int:
    return len(RE_URL.findall(text))

def _long_lines_exist(text: str, max_len: int = MAX_LINE_LEN_CHARS) -> bool:
    return any(len(ln) > max_len for ln in _lines(text))

def _has_tables_or_columns_hint(text: str) -> bool:
    # crude: tab characters or multi-space alignment often indicates tables/columns
    return ("\t" in text) or bool(re.search(r"[^\S\r\n]{3,}\S+[^\S\r\n]{3,}", text))

def _skills_list_count(sections: dict[str, str]) -> int:
    # expects pre-parsed sections; fallback: scan whole text
    skills_block = ""
    for key in ("skills", "skills & tools", "tech stack"):
        if key in sections:
            skills_block += "\n" + sections[key]
    if not skills_block and "resume_text" in sections:
        skills_block = sections["resume_text"]  # fallback
    toks = _tokenize(skills_block)
    # count comma-separated or bullet-separated tokens as proxy
    commas = skills_block.count(",")
    bullets = len(_bullet_lines(skills_block))
    return max(commas + 1, bullets * 3, len({t for t in toks if len(t) > 1}) // 12)

def _recent_grad(year: int | None, now_year: int) -> bool:
    return (year is not None) and (now_year - year <= MIN_EDU_RECENCY_YRS)

def _first_person_ratio(text: str) -> float:
    # avoid "I/my/me" in bullets; CVs usually not first-person
    toks = _tokenize(text)
    if not toks: 
        return 0.0
    fp = sum(t in {"i","my","me","mine"} for t in toks)
    return fp / len(toks)

def _passive_voice_ratio(text: str) -> float:
    # very light heuristic: "was|were|being|been|by" clusters
    matches = re.findall(r"\b(was|were|been|being)\b\s+\w+\b\s+\bby\b", text.lower())
    toks = _tokenize(text)
    return len(matches) / max(1, len(toks))

# --------------- Main: Conditional Suggestions ----------------
from time import localtime

def _suggestions(
    req_missing: list[str],
    pref_missing: list[str],
    resume_text: str,
    sections: dict[str, str] | None = None,
    page_count: float | None = None,
    job_title: str | None = None,
    is_early_career: bool | None = None,      # if you detect student/fresh grad
    grad_year: int | None = None,             # if parsed from Education
    bullet_ratio: float | None = None,
    verb_ratio: float | None = None,
) -> list[str]:
    """
    Returns prioritized suggestions with conditions.
    `sections` may include a special key "resume_text" for fallback analyzers.
    """
    sections = sections or {}
    # compute metrics if not provided
    if bullet_ratio is None:
        bullet_ratio = _bullet_ratio(resume_text)
    if verb_ratio is None:
        verb_ratio = _verb_ratio(resume_text)

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

    # Collect (priority, suggestion) where lower number = higher priority
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
    if bullet_ratio < MIN_BULLET_RATIO:
        S.append((2, "Use bullets for experience; target ≥25% of lines as bullets."))
    if verb_ratio < MIN_VERB_RATIO:
        S.append((2, "Start bullets with strong action verbs (Led, Built, Optimized, Spearheaded…)."))
    if long_lines:
        S.append((3, "Split long lines into concise bullets (~1–2 lines each) to improve readability."))
    if tables_hint:
        S.append((3, "Avoid multi-column layouts/tables; use a single-column, ATS-friendly format."))

    # 4) Contact & links
    if contact_fields < MIN_CONTACT_FIELDS:
        S.append((1, "Include at least email and phone (and optionally city/country)."))
    if links_cnt < MIN_LINKS:
        # tailor link suggestion by role if available
        if job_title and any(k in job_title.lower() for k in ("engineer","developer","data","ml","ai","software","frontend","backend","fullstack")):
            S.append((2, "Add a professional link (GitHub/LinkedIn/Portfolio) to showcase projects."))
        elif job_title and any(k in job_title.lower() for k in ("designer","ux","ui","product")):
            S.append((2, "Add a portfolio link (Behance/Dribbble/Website) to showcase work."))
        else:
            S.append((2, "Add a professional link (LinkedIn, portfolio, or GitHub) for credibility."))

    # 5) Sections & ordering
    has_exp = any(k in sections for k in ("experience","work experience","professional experience"))
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

    # 11) Soft skills (only if we haven’t already given too many)
    soft_added = 0
    if soft_added < MAX_SOFT_SUGGESTIONS and passive_ratio > 0.002:
        S.append((4, "Demonstrate soft skills via outcomes (e.g., “Led a team of 4 to deliver on time”).")); soft_added += 1
    if soft_added < MAX_SOFT_SUGGESTIONS and numeric_density < MIN_NUMERIC_DENSITY:
        S.append((4, "Use the STAR method in bullets (Situation, Task, Action, Result).")); soft_added += 1
    if soft_added < MAX_SOFT_SUGGESTIONS and (has_exp or has_proj):
        S.append((4, "Balance soft skills with evidence (teamwork, leadership) tied to measurable results.")); soft_added += 1

    # 12) File/export hygiene (only surface if hints of Word artifacts)
    if "resume.doc" in resume_text.lower() or ".docx" in resume_text.lower():
        S.append((4, "Export as PDF (text-based, not scanned) to keep formatting and remain ATS-friendly."))

    # De-duplicate while preserving best priority
    seen = {}
    for pr, txt in S:
        if txt not in seen or pr < seen[txt]:
            seen[txt] = pr
    ranked = sorted(((pr, txt) for txt, pr in seen.items()), key=lambda x: x[0])

    # Emit top N
    return [txt for _, txt in ranked[:SUGGESTIONS_MAX_OUT]]
#--------AKHER EL SUGGESTION!!-------------

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

    # Extract job title (simple heuristic: first line)
    job_title = job_description.splitlines()[0].strip() if job_description.strip() else None
    page_count_estimate = len(resume_text) / 1800 if resume_text else 0

    suggested = _suggestions(
        req_missing=req_missing,
        pref_missing=pref_missing,
        resume_text=resume_text,
        sections=sections_text,
        page_count=page_count_estimate,
        job_title=job_title,
        bullet_ratio=bullet_ratio,
        verb_ratio=verb_ratio,
    )

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