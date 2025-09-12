import os
import json
import uuid
import time
import hmac, hashlib, base64
from io import BytesIO
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from passlib.context import CryptContext

from reportlab.platypus import Table
from reportlab.lib import colors
from ats_core import compute_score, _maybe_load_model

# ---------- Optional deps (PDF, parsing) ----------
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _PDF_OK = True
except Exception:
    _PDF_OK = False

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    from pdfminer.layout import LAParams
except ImportError:
    pdf_extract_text = None
    LAParams = None

def read_pdf_bytes(b: bytes) -> str:
    if pdf_extract_text is None or LAParams is None:
        raise HTTPException(500, "PDF parsing requires 'pdfminer.six'. Install it.")
    laparams = LAParams()  # tune if needed (e.g., line_margin, word_margin)
    return pdf_extract_text(BytesIO(b), laparams=laparams) or ""

try:
    import docx  # python-docx
except ImportError:
    docx = None

# ---------- Paths & config ----------
HERE         = Path(__file__).resolve().parent
INDEX_PATH   = HERE / "index.html"
LOGIN_PATH   = HERE / "login.html"
REGISTER_PATH= HERE / "register.html"
REPORT_DIR   = HERE / "reports"
HOME_PATH = HERE / "home.html"
REPORT_DIR.mkdir(exist_ok=True)

APP_TITLE = "ATS-like Resume Checker API"
VERSION   = "0.3.2"

# Upload limits
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "8"))
MAX_BYTES     = int(MAX_UPLOAD_MB * 1024 * 1024)

# Rate limit (req/min per IP)
MAX_RPM = int(os.environ.get("MAX_RPM", "120"))
_REQ_LOG: dict[str, list[float]] = {}

# Auth config (for local HTTP set COOKIE_SECURE=0)
LOGIN_ENABLED  = os.environ.get("LOGIN_ENABLED", "1") == "1"
DEMO_USER      = os.environ.get("DEMO_USER", "demo")
DEMO_PASS      = os.environ.get("DEMO_PASS", "letmein")
SESSION_COOKIE = "ats_session"
COOKIE_SECURE  = bool(int(os.environ.get("COOKIE_SECURE", "0")))  # default 0 for local dev
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")             # set a strong secret in prod

# DB (SQLite by default)
DB_URL = os.environ.get("DB_URL", f"sqlite:///{HERE / 'users.db'}")
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, h: str) -> bool:
    return pwd_context.verify(p, h)

# ---------- Models ----------
class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        print("DB init failed:", e)
    try:
        if os.environ.get("EMB_ON", "0") == "1":
            _maybe_load_model()
            print("Embeddings model preloaded.")
    except Exception as e:
        print("Embeddings preload skipped/failed:", e)
    yield

# ---------- Create app ----------
app = FastAPI(title=APP_TITLE, version=VERSION, lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def rate_limit(request: Request) -> None:
    now = time.time()
    ip = request.client.host if request and request.client else "unknown"
    q = [t for t in _REQ_LOG.get(ip, []) if now - t < 60]
    if len(q) >= MAX_RPM:
        raise HTTPException(429, "Too many requests, slow down.")
    q.append(now)
    _REQ_LOG[ip] = q

def _make_token(username: str) -> str:
    if not SESSION_SECRET:
        return "ok"
    payload = (username or "user").encode("utf-8")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    data = base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
    return f"{data}.{sig}"

def _verify_token(token: str) -> bool:
    if not SESSION_SECRET:
        return token == "ok"
    try:
        data, sig = token.split(".", 1)
        payload = base64.urlsafe_b64decode(data + "===")  # pad
        expected = hmac.new(SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

def is_authed(request: Request) -> bool:
    if not LOGIN_ENABLED:
        return True
    tok = request.cookies.get(SESSION_COOKIE)
    return bool(tok) and _verify_token(tok)

def read_docx_bytes(b: bytes) -> str:
    if docx is None:
        raise HTTPException(500, "DOCX parsing requires 'python-docx'. Install it.")
    d = docx.Document(BytesIO(b))
    return "\n".join(p.text for p in d.paragraphs)

# at top of app.py (or a pdf_utils.py)
import io

# Optional imports guarded (install when you can)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
except Exception:
    pdfminer_extract_text, LAParams = None, None

try:
    import pytesseract
    from pdf2image import convert_from_bytes
except Exception:
    pytesseract = None
    convert_from_bytes = None


def _extract_pdf_text_pdfminer(b: bytes) -> str:
    if pdfminer_extract_text is None:
        return ""
    laparams = LAParams()  # you can tune: line_margin, word_margin
    try:
        return pdfminer_extract_text(io.BytesIO(b), laparams=laparams) or ""
    except Exception:
        return ""


def _extract_pdf_text_pymupdf_blocks(b: bytes) -> str:
    """
    Use PyMuPDF to read text blocks with coordinates and reconstruct left/right columns.
    This is MUCH more reliable for Canva and design-tool PDFs.
    """
    if fitz is None:
        return ""

    try:
        doc = fitz.open(stream=b, filetype="pdf")
    except Exception:
        return ""

    all_pages_text = []

    for page in doc:
        # get blocks: [x0, y0, x1, y1, "text", block_no, block_type, ...]
        # Using get_text("blocks") returns tuples; order is arbitrary → we sort.
        blocks = page.get_text("blocks") or []
        if not blocks:
            continue

        # Sort by x, then y for stable grouping
        blocks_sorted = sorted(blocks, key=lambda bl: (round(bl[0], 1), round(bl[1], 1)))

        # Find a rough vertical mid to split columns
        xs = [bl[0] for bl in blocks_sorted]
        if not xs:
            continue
        x_min, x_max = min(xs), max(xs)
        mid = (x_min + x_max) / 2.0

        left_blocks  = [bl for bl in blocks_sorted if bl[0] < mid]
        right_blocks = [bl for bl in blocks_sorted if bl[0] >= mid]

        # Within each column, sort by (y, x) reading order
        left_blocks  = sorted(left_blocks,  key=lambda bl: (round(bl[1], 1), round(bl[0], 1)))
        right_blocks = sorted(right_blocks, key=lambda bl: (round(bl[1], 1), round(bl[0], 1)))

        def _join_blocks(col_blocks):
            texts = []
            for bl in col_blocks:
                t = bl[4] if len(bl) > 4 else ""
                if t:
                    texts.append(t.strip())
            # keep column separation with double newline
            return "\n".join([t for t in texts if t])

        # Important: Left column often holds headers; right is main content.
        left_text  = _join_blocks(left_blocks)
        right_text = _join_blocks(right_blocks)

        page_text = ""
        if left_text.strip():
            page_text += left_text.strip() + "\n"
        if right_text.strip():
            page_text += right_text.strip()
        if page_text.strip():
            all_pages_text.append(page_text)

    return "\n\n".join(all_pages_text).strip()


def _extract_pdf_text_ocr(b: bytes) -> str:
    """
    OCR fallback for image-only PDFs.
    Requires: pytesseract, pdf2image + poppler (system)
    """
    if pytesseract is None or convert_from_bytes is None:
        return ""

    try:
        images = convert_from_bytes(b, dpi=300)
    except Exception:
        return ""

    ocr_texts = []
    for img in images:
        try:
            txt = pytesseract.image_to_string(img)
        except Exception:
            txt = ""
        if txt:
            ocr_texts.append(txt)

    return "\n".join(ocr_texts).strip()


def read_pdf_bytes(b: bytes) -> str:
    """
    Robust extractor that tries:
      1) PyMuPDF blocks (layout/columns aware)
      2) pdfminer (stream text)
      3) OCR fallback (image-only PDFs like some Canva exports)
    """
    # 1) Try layout-aware first (best for Canva)
    text = _extract_pdf_text_pymupdf_blocks(b)
    if text and len(text.strip()) >= 20:
        return text

    # 2) Try pdfminer
    text = _extract_pdf_text_pdfminer(b)
    if text and len(text.strip()) >= 20:
        return text

    # 3) If still nothing useful, OCR
    text = _extract_pdf_text_ocr(b)
    return text or ""

def decode_text_bytes(b: bytes) -> str:
    return b.decode("utf-8", errors="ignore")

def docx_stats_from_bytes(b: bytes) -> dict:
    stats = {"tables": 0, "images": 0}
    try:
        if docx is None:
            return stats
        d = docx.Document(BytesIO(b))
        stats["tables"] = len(getattr(d, "tables", []) or [])
        stats["images"] = len(getattr(d, "inline_shapes", []))
    except Exception:
        pass
    return stats

# PDF
def _canonical_section_rows(secs: dict) -> list[list[str]]:
    """Collapse alias families so we show each section once, cleanly."""
    yes = lambda b: "Yes" if b else "No"

    experience = bool(
        secs.get("experience")
        or secs.get("work experience")
        or secs.get("professional experience")
    )
    skills = bool(
        secs.get("skills")
        or secs.get("skills & tools")
        or secs.get("tech stack")
    )
    summary = bool(
        secs.get("summary")
        or secs.get("professional summary")
        or secs.get("profile")
        or secs.get("profile summary")
        or secs.get("objective")
    )
    projects = bool(secs.get("projects"))
    education = bool(secs.get("education"))
    certifications = bool(secs.get("certifications"))
    publications = bool(secs.get("publications"))

    # Only canonical rows; aliases are intentionally omitted.
    rows = [
        ["Experience",     yes(experience)],
        ["Projects",       yes(projects)],
        ["Education",      yes(education)],
        ["Skills",         yes(skills)],
        ["Certifications", yes(certifications)],
        ["Publications",   yes(publications)],
        ["Summary",        yes(summary)],
    ]
    return rows

_PDF_UNICODE = False
_PDF_FONT_NAME = "UIUnicode"

def _pdf_safe_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    repl = {"•": "-", "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "≥": ">=", "≤": "<=", "×": "x",
            "™": "(TM)", "©": "(C)", "®": "(R)", "→": "->", "←": "<-"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1", "ignore")

def _pdf_text(s: str) -> str:
    return s if _PDF_UNICODE else _pdf_safe_text(s)

def _ensure_pdf_font():
    global _PDF_UNICODE
    if not _PDF_OK or _PDF_UNICODE:
        return
    try_paths = []
    if os.name == "nt":
        try_paths += [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\Arial.ttf", r"C:\Windows\Fonts\Calibri.ttf"]
    else:
        try_paths += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for fp in try_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont(_PDF_FONT_NAME, fp))
                _PDF_UNICODE = True
                break
            except Exception:
                continue

def build_pdf(report: dict) -> bytes:
    if not _PDF_OK:
        raise HTTPException(500, "PDF export requires 'reportlab'. Install it")
    _ensure_pdf_font()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="ATS Resume Check Report")
    styles = getSampleStyleSheet()
    if _PDF_UNICODE:
        for k in ("Heading1", "Heading2", "Heading3", "BodyText"):
            styles[k].fontName = _PDF_FONT_NAME
    H1, H2, H3, P = styles["Heading1"], styles["Heading2"], styles["Heading3"], styles["BodyText"]

    def kv_table(d: dict, keys: list[str]):
        rows = [["Metric", "Value"]]
        for k in keys:
            v = d.get(k, "")
            rows.append([k.replace("_", " ").title(), str(v)])
        t = Table(rows, colWidths=[200, 320])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        return t

    # collapse aliases → single canonical rows
    def _canonical_section_rows(secs: dict) -> list[list[str]]:
        yn = lambda b: "Yes" if b else "No"

        experience = bool(
            secs.get("experience")
            or secs.get("work experience")
            or secs.get("professional experience")
        )
        skills = bool(
            secs.get("skills")
            or secs.get("skills & tools")
            or secs.get("tech stack")
        )
        summary = bool(
            secs.get("summary")
            or secs.get("professional summary")
            or secs.get("profile")
            or secs.get("profile summary")
            or secs.get("objective")
        )
        projects       = bool(secs.get("projects"))
        education      = bool(secs.get("education"))
        certifications = bool(secs.get("certifications"))
        publications   = bool(secs.get("publications"))

        return [
            ["Experience",     yn(experience)],
            ["Projects",       yn(projects)],
            ["Education",      yn(education)],
            ["Skills",         yn(skills)],
            ["Certifications", yn(certifications)],
            ["Publications",   yn(publications)],
            ["Summary",        yn(summary)],
        ]

    story = []
    story.append(Paragraph("ATS Resume Check Report", H1))
    story.append(Paragraph(f"Score: {report.get('score', 0)} / 100", H2))
    story.append(Spacer(1, 12))

    comps = report.get("components", {}) or {}
    comp_keys = [
        "required_coverage", "overall_coverage",
        "frequency_bonus_required", "frequency_bonus_overall",
        "section_hygiene", "bullets_ratio_%", "action_verb_ratio_%"
    ]
    story.append(Paragraph("Components", H2))
    story.append(kv_table(comps, comp_keys))
    story.append(Spacer(1, 12))

    def bullets(title: str, items: list[str]):
        story.append(Paragraph(title, H3))
        if not items:
            story.append(Paragraph("None", P))
            story.append(Spacer(1, 6))
            return
        for it in items:
            story.append(Paragraph(f"- {it}", P))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 6))
    bullets("Matched Keywords", report.get("matched_keywords") or [])
    bullets("Missing (Required)", report.get("required_missing") or [])
    bullets("Missing (Preferred)", report.get("preferred_missing") or [])
    bullets("Suggestions", report.get("suggested_bullets") or [])

    # Section Presence (canonicalized, no duplicate alias rows)
    secs = report.get("section_presence", {}) or {}
    if secs:
        rows = [["Section", "Present?"]] + _canonical_section_rows(secs)
        t = Table(rows, colWidths=[260, 260])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(Paragraph("Section Presence", H2))
        story.append(t)
        story.append(Spacer(1, 12))

    warns = report.get("format_warnings") or []
    if warns:
        bullets("Parser-Friendliness Tips", warns)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

def build_warnings(report: dict, *, ext: Optional[str], r_bytes: Optional[bytes], docx_stats: Optional[dict]) -> list[str]:
    w: list[str] = []
    comp = report.get("components", {}) or {}
    secs = report.get("section_presence", {}) or {}
    need = ["experience", "skills", "projects", "education"]
    missing_secs = [s for s in need if not secs.get(s, False)]
    if missing_secs:
        w.append("Add standard headings: " + ", ".join(s.upper() for s in missing_secs) + ".")
    if comp.get("bullets_ratio_%", 0) < 25:
        w.append("Use bullets for experience; aim for ≥25% of lines as bullets.")
    if comp.get("action_verb_ratio_%", 0) < 15:
        w.append("Start more bullets with action verbs (Built, Implemented, Optimized…).")
    if ext == "pdf" and r_bytes is not None and not report.get("top_resume_terms"):
        w.append("Your PDF may be image-only. Use a selectable-text PDF or DOCX, or run OCR.")
    if ext == "docx" and docx_stats:
        if (docx_stats.get("tables") or 0) > 0:
            w.append("Avoid complex tables—ATS parsers may skip table content.")
        if (docx_stats.get("images") or 0) > 0:
            w.append("Don’t put text inside images; parsers can’t read it.")
    return w

# ---------- Routes ----------
@app.get("/", include_in_schema=False)
def root_redirect():
    # If you prefer to keep it protected + JSON, delete this and keep your old root() handler.
    return RedirectResponse(url="/home")

@app.get("/home", include_in_schema=False)
def home_page(request: Request):
    if not HOME_PATH.exists():
        raise HTTPException(404, "home.html not found next to app.py")
    return FileResponse(HOME_PATH)

@app.get("/login", include_in_schema=False)
def login_page():
    if not LOGIN_PATH.exists():
        raise HTTPException(404, "login.html not found next to app.py")
    return FileResponse(LOGIN_PATH)

@app.get("/register", include_in_schema=False)
def register_page():
    if not REGISTER_PATH.exists():
        raise HTTPException(404, "register.html not found next to app.py")
    return FileResponse(REGISTER_PATH)

@app.post("/auth/register")
async def auth_register(username: str = Form(...), password: str = Form(...)):
    if not LOGIN_ENABLED:
        raise HTTPException(400, "Registration disabled in this environment.")
    u = username.strip().lower()
    if not (3 <= len(u) <= 32):
        raise HTTPException(400, "Username must be 3–32 characters.")
    if not all(ch.isalnum() or ch in "._" for ch in u):
        raise HTTPException(400, "Username may contain letters, numbers, dot, underscore.")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    db = SessionLocal()
    try:
        if db.query(User).filter_by(username=u).first():
            raise HTTPException(400, "Username already exists.")
        db.add(User(username=u, password_hash=hash_password(password)))
        db.commit()
        return {"ok": True, "message": "Account created. You can sign in now."}
    finally:
        db.close()

@app.post("/auth/login")
async def auth_login(response: Response, username: str = Form(...), password: str = Form(...)):
    if not LOGIN_ENABLED:
        return {"ok": True, "redirect": "/ui"}
    u = username.strip().lower()
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=u).first()
    finally:
        db.close()
    ok = False
    if user and verify_password(password, user.password_hash):
        ok = True
    elif u == DEMO_USER and password == DEMO_PASS:
        ok = True
    if not ok:
        raise HTTPException(401, "Invalid credentials")
    tok = _make_token(u)
    resp = JSONResponse({"ok": True, "redirect": "/ui"})
    resp.set_cookie(SESSION_COOKIE, tok, httponly=True, samesite="lax", secure=COOKIE_SECURE, max_age=60*60*24*7)
    return resp

@app.post("/auth/logout")
def auth_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp

@app.get("/health", include_in_schema=False)
def health(request: Request):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    return {"ok": True, "version": VERSION}

@app.get("/config", include_in_schema=False)
def config(request: Request):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    return {"version": VERSION, "max_upload_mb": MAX_UPLOAD_MB, "max_rpm": MAX_RPM}



@app.get("/ui", include_in_schema=False)
def ui(request: Request):
    if not is_authed(request):
        return RedirectResponse(url="/login")
    if not INDEX_PATH.exists():
        raise HTTPException(404, "index.html not found next to app.py")  # <- was ats_app.py
    return FileResponse(INDEX_PATH)

# Reports (open for sharing)
@app.get("/r/{rid}")
def get_report(rid: str):
    p = REPORT_DIR / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "Report not found")
    data = json.loads(p.read_text("utf-8"))
    data.setdefault("id", rid)
    data.setdefault("share_url", f"/r/{rid}")
    return JSONResponse(data)

@app.get("/report/{rid}.pdf")
def report_pdf(rid: str):
    p = REPORT_DIR / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "Report not found")
    report = json.loads(p.read_text("utf-8"))
    pdf_bytes = build_pdf(report)
    filename = f"ats_report_{rid}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)

# ---------- Scoring ----------
@app.post("/score")
def score(req: ScoreRequest, request: Request):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    rate_limit(request)
    report = compute_score(req.resume_text, req.job_description)
    report["format_warnings"] = build_warnings(report, ext=None, r_bytes=None, docx_stats=None)
    rid = str(uuid.uuid4())
    (REPORT_DIR / f"{rid}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    report["id"] = rid
    report["share_url"] = f"/r/{rid}"
    return report

@app.post("/score-file")
async def score_file(
    request: Request,
    resume: UploadFile = File(...),
    job_description: UploadFile | None = File(None),
    jd_text: str | None = Form(None),
):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    rate_limit(request)

    r_bytes = await resume.read()
    if len(r_bytes) > MAX_BYTES:
        raise HTTPException(413, f"Resume file too large (>{MAX_UPLOAD_MB} MB).")

    r_ext = (resume.filename or "").lower().split(".")[-1]
    if r_ext == "pdf":
        resume_text = read_pdf_bytes(r_bytes)
    elif r_ext == "docx":
        resume_text = read_docx_bytes(r_bytes)
    else:
        resume_text = decode_text_bytes(r_bytes)

    if job_description is not None:
        jd_bytes = await job_description.read()
        if len(jd_bytes) > MAX_BYTES:
            raise HTTPException(413, f"Job description file too large (>{MAX_UPLOAD_MB} MB).")
        jd_ext = (job_description.filename or "").lower().split(".")[-1]
        if jd_ext == "pdf":
            jd_text = read_pdf_bytes(jd_bytes)
        elif jd_ext == "docx":
            jd_text = read_docx_bytes(jd_bytes)
        else:
            jd_text = decode_text_bytes(jd_bytes)

    if not jd_text or not jd_text.strip():
        raise HTTPException(400, "Provide job_description file or jd_text.")
    if not resume_text.strip():
        raise HTTPException(400, "No text found in resume. If it’s a scanned PDF, convert to DOCX or use OCR.")

    report = compute_score(resume_text, jd_text)
    docx_meta = docx_stats_from_bytes(r_bytes) if r_ext == "docx" else None
    report["format_warnings"] = build_warnings(report, ext=r_ext, r_bytes=r_bytes, docx_stats=docx_meta)

    rid = str(uuid.uuid4())
    (REPORT_DIR / f"{rid}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    report["id"] = rid
    report["share_url"] = f"/r/{rid}"
    return report
