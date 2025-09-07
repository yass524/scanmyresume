# app.py
# FastAPI backend for ATS Checker
# - UI (/ui) behind optional login (/login)
# - Score endpoints: /score, /score-file
# - Saved reports: /r/{id}, PDF: /report/{id}.pdf
# - Health/config, rate limiting, parser-friendliness audit
# - Embeddings preload (when EMB_ON=1)
#sherif
import os
import json
import uuid
import time
import hmac, hashlib, base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from ats_core import compute_score  # and _emb_model (lazy-imported in startup)

# ---------- Optional deps ----------
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    # import pdfbase only if reportlab present
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _PDF_OK = True
except Exception:
    _PDF_OK = False

try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # pip install pdfminer.six
except ImportError:
    pdf_extract_text = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

# ---------- App & config ----------
APP_TITLE = "ATS-like Resume Checker API"
VERSION = "0.3.1"

app = FastAPI(title=APP_TITLE, version=VERSION)

# CORS (open in dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Paths & storage
HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / "index.html"
LOGIN_PATH = HERE / "login.html"
REPORT_DIR = HERE / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Upload limits
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "8"))
MAX_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# Rate limit (requests/min per IP)
MAX_RPM = int(os.environ.get("MAX_RPM", "120"))
_REQ_LOG: dict[str, list[float]] = {}

# Auth config
LOGIN_ENABLED  = os.environ.get("LOGIN_ENABLED", "1") == "1"
DEMO_USER      = os.environ.get("DEMO_USER", "demo")
DEMO_PASS      = os.environ.get("DEMO_PASS", "letmein")
SESSION_COOKIE = "ats_session"
COOKIE_SECURE  = bool(int(os.environ.get("COOKIE_SECURE", "1")))  # set 0 for http dev
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")             # set strong secret in prod

# ---------- Models ----------
class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str

# ---------- Utils ----------
def rate_limit(request: Request) -> None:
    now = time.time()
    ip = request.client.host if request and request.client else "unknown"
    q = [t for t in _REQ_LOG.get(ip, []) if now - t < 60]
    if len(q) >= MAX_RPM:
        raise HTTPException(429, "Too many requests, slow down.")
    q.append(now)
    _REQ_LOG[ip] = q

def save_report(report: dict) -> str:
    rid = str(uuid.uuid4())
    report["id"] = rid
    report["share_url"] = f"/r/{rid}"
    (REPORT_DIR / f"{rid}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    return rid


def read_docx_bytes(b: bytes) -> str:
    if docx is None:
        raise HTTPException(500, "DOCX parsing requires 'python-docx'. Install it.")
    d = docx.Document(BytesIO(b))
    return "\n".join(p.text for p in d.paragraphs)

def read_pdf_bytes(b: bytes) -> str:
    if pdf_extract_text is None:
        raise HTTPException(500, "PDF parsing requires 'pdfminer.six'. Install it.")
    return pdf_extract_text(BytesIO(b)) or ""

def decode_text_bytes(b: bytes) -> str:
    return b.decode("utf-8", errors="ignore")

# ----- Parser-friendliness helpers -----
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

def build_warnings(
    report: dict, *, ext: Optional[str], r_bytes: Optional[bytes], docx_stats: Optional[dict]
) -> list[str]:
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

# ---------- PDF text safety / fonts ----------
_PDF_UNICODE = False  # becomes True if we register a unicode TTF
_PDF_FONT_NAME = "UIUnicode"

def _pdf_safe_text(s: str) -> str:
    """Map common unicode to ASCII; final guard drops non-latin-1."""
    if not isinstance(s, str):
        s = str(s)
    repl = {
        "•": "-", "–": "-", "—": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "…": "...", "≥": ">=", "≤": "<=", "×": "x",
        "™": "(TM)", "©": "(C)", "®": "(R)",
        "→": "->", "←": "<-", "⇄": "<->",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1", "ignore")

def _pdf_text(s: str) -> str:
    return s if _PDF_UNICODE else _pdf_safe_text(s)

def _ensure_pdf_font():
    """Try to register a Unicode font once (Arial on Windows, DejaVuSans on Linux)."""
    global _PDF_UNICODE
    if not _PDF_OK or _PDF_UNICODE:
        return
    try_paths = []
    if os.name == "nt":
        try_paths += [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\Arial.ttf",
            r"C:\Windows\Fonts\Calibri.ttf",
        ]
    else:
        try_paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for fp in try_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont(_PDF_FONT_NAME, fp))
                _PDF_UNICODE = True
                break
            except Exception:
                continue

# ----- PDF builder -----
def build_pdf(report: dict) -> bytes:
    if not _PDF_OK:
        raise HTTPException(500, "PDF export requires 'reportlab'. Install it with: pip install reportlab")

    _ensure_pdf_font()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="ATS Resume Check Report")
    styles = getSampleStyleSheet()
    if _PDF_UNICODE:
        styles["Heading1"].fontName = _PDF_FONT_NAME
        styles["Heading2"].fontName = _PDF_FONT_NAME
        styles["Heading3"].fontName = _PDF_FONT_NAME
        styles["BodyText"].fontName = _PDF_FONT_NAME
    H1, H2, H3 = styles["Heading1"], styles["Heading2"], styles["Heading3"]
    P = styles["BodyText"]

    def kv_table(d: dict, keys: list[str]) -> Table:
        rows = [[_pdf_text("Metric"), _pdf_text("Value")]]
        for k in keys:
            v = d.get(k, "")
            rows.append([_pdf_text(k.replace("_", " ").title()), _pdf_text(str(v))])
        t = Table(rows, colWidths=[200, 320])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#0b1220"), colors.HexColor("#0f172a")]),
            ("TEXTCOLOR",  (0,1), (-1,-1), colors.whitesmoke),
            ("INNERGRID",  (0,0), (-1,-1), 0.25, colors.grey),
            ("BOX",        (0,0), (-1,-1), 0.5, colors.grey),
            ("LEFTPADDING",(0,0), (-1,-1), 6), ("RIGHTPADDING",(0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        return t

    bullet_char = "•" if _PDF_UNICODE else "-"

    def bullets(title: str, items: list[str]):
        flow = [Paragraph(_pdf_text(title), H3)]
        if not items:
            flow.append(Paragraph(_pdf_text("None"), P))
            flow.append(Spacer(1, 6))
            return flow
        for it in items:
            flow.append(Paragraph(_pdf_text(f"{bullet_char} {it}"), P))
        flow.append(Spacer(1, 6))
        return flow

    comps = report.get("components", {}) or {}
    secs = report.get("section_presence", {}) or {}
    gaps_by_cat = report.get("gaps_by_category", {}) or {}

    story = []
    story.append(Paragraph(_pdf_text("ATS Resume Check Report"), H1))
    story.append(Paragraph(_pdf_text(f"Score: {report.get('score', 0)} / 100"), H2))
    story.append(Spacer(1, 12))

    comp_keys = [
        "required_coverage","overall_coverage",
        "frequency_bonus_required","frequency_bonus_overall",
        "section_hygiene","bullets_ratio_%","action_verb_ratio_%",
    ]
    story.append(Paragraph(_pdf_text("Components"), H2))
    story.append(kv_table(comps, comp_keys))
    story.append(Spacer(1, 12))

    story += bullets("Matched Keywords", report.get("matched_keywords") or [])
    story += bullets("Missing (Required)", report.get("required_missing") or [])
    story += bullets("Missing (Preferred)", report.get("preferred_missing") or [])
    story += bullets("Suggestions", report.get("suggested_bullets") or [])
    story.append(Spacer(1, 6))

    sec_rows = [[ _pdf_text("Section"), _pdf_text("Present?") ]]
    for k, v in secs.items():
        sec_rows.append([ _pdf_text(k.title()), _pdf_text("Yes" if v else "No") ])
    if len(sec_rows) > 1:
        t_secs = Table(sec_rows, colWidths=[260, 260])
        t_secs.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#0b1220"), colors.HexColor("#0f172a")]),
            ("TEXTCOLOR",  (0,1), (-1,-1), colors.whitesmoke),
            ("INNERGRID",  (0,0), (-1,-1), 0.25, colors.grey),
            ("BOX",        (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(Paragraph(_pdf_text("Section Presence"), H2))
        story.append(t_secs)
        story.append(Spacer(1, 12))

    if gaps_by_cat:
        cat_rows = [[ _pdf_text("Category"), _pdf_text("Terms") ]]
        for cat, terms in gaps_by_cat.items():
            cat_rows.append([ _pdf_text(cat), _pdf_text(", ".join(terms or [])) ])
        t_cat = Table(cat_rows, colWidths=[140, 380])
        t_cat.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#0b1220"), colors.HexColor("#0f172a")]),
            ("TEXTCOLOR",  (0,1), (-1,-1), colors.whitesmoke),
            ("INNERGRID",  (0,0), (-1,-1), 0.25, colors.grey),
            ("BOX",        (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(Paragraph(_pdf_text("Gaps by Category"), H2))
        story.append(t_cat)
        story.append(Spacer(1, 12))

    warns = report.get("format_warnings") or []
    if warns:
        story += bullets("Parser-Friendliness Tips", warns)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

# ----- Auth helpers -----
def _make_token(username: str) -> str:
    """If SESSION_SECRET is set, return HMAC token: <b64(username)>.<sig>; else 'ok' (dev)."""
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

# ---------- Startup ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # preload embeddings if enabled
    try:
        if os.environ.get("EMB_ON", "0") == "1":
            from ats_core import _emb_model
            _ = _emb_model
    except Exception as e:
        print("Embeddings preload skipped/failed:", e)
    yield  # (place shutdown/cleanup after yield if you add any)

# pass lifespan when creating the app
app = FastAPI(title=APP_TITLE, version=VERSION, lifespan=lifespan)

# ---------- Routes ----------
@app.get("/login", include_in_schema=False)
def login_page():
    if not LOGIN_PATH.exists():
        raise HTTPException(404, "login.html not found next to app.py")
    return FileResponse(LOGIN_PATH)

@app.post("/auth/login")
async def auth_login(response: Response, username: str = Form(...), password: str = Form(...)):
    if not LOGIN_ENABLED:
        return {"ok": True, "redirect": "/ui"}
    if username == DEMO_USER and password == DEMO_PASS:
        tok = _make_token(username)
        resp = JSONResponse({"ok": True, "redirect": "/ui"})
        resp.set_cookie(
            SESSION_COOKIE, tok,
            httponly=True, samesite="lax",
            secure=COOKIE_SECURE, max_age=60*60*24*7  # 7 days
        )
        return resp
    raise HTTPException(401, "Invalid credentials")

@app.post("/auth/logout")
def auth_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "version": VERSION}

@app.get("/config", include_in_schema=False)
def config():
    return {"version": VERSION, "max_upload_mb": MAX_UPLOAD_MB, "max_rpm": MAX_RPM}

@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "try": "/docs", "ui": "/ui", "post": "/score"}

@app.get("/ui", include_in_schema=False)
def ui(request: Request):
    if not is_authed(request):
        return RedirectResponse(url="/login")
    if not INDEX_PATH.exists():
        raise HTTPException(404, "index.html not found next to app.py")
    return FileResponse(INDEX_PATH)

@app.get("/r/{rid}")
def get_report(rid: str):
    p = REPORT_DIR / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "Report not found")
    data = json.loads(p.read_text("utf-8"))
    # Backfill for older saved reports
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

# ---------- Scoring endpoints ----------
@app.post("/score")
def score(req: ScoreRequest, request: Request):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    rate_limit(request)

    report = compute_score(req.resume_text, req.job_description)
    report["format_warnings"] = build_warnings(report, ext=None, r_bytes=None, docx_stats=None)

    rid = save_report(report)
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

    rid = save_report(report)
    report["id"] = rid
    report["share_url"] = f"/r/{rid}"
    return report
