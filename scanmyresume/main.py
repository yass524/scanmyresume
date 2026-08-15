import json
import os
import uuid
import re
import secrets
import hashlib
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from scanmyresume.ats.core import _maybe_load_model, compute_score
from scanmyresume.auth import is_authed, make_token, rate_limit
from scanmyresume.config import (
    ADS_FALLBACK,
    ADS_PATH,
    APP_BASE_URL,
    APP_TITLE,
    COOKIE_SECURE,
    FORGOT_PASSWORD_PATH,
    HOME_PATH,
    INDEX_PATH,
    LOGIN_ENABLED,
    LOGIN_PATH,
    REGISTER_PATH,
    REPORT_DIR,
    RESET_PASSWORD_PATH,
    RESET_TOKEN_TTL_MINUTES,
    SESSION_COOKIE,
    VERSION,
)
from scanmyresume.database import Base, PasswordResetToken, SessionLocal, User, engine, hash_password, verify_password
from scanmyresume.services.ai_grading import apply_ai_grading
from scanmyresume.services.email import send_email
from scanmyresume.services.files import (
    docx_stats_from_bytes,
    ensure_size,
    ext_from_filename,
    read_docx_bytes,
    read_pdf_bytes,
    sanitize_unicode,
)
from scanmyresume.services.jd_quality import assess_jd_quality, validate_jd_hard
from scanmyresume.services.pdf_reports import build_pdf_from_report
from scanmyresume.services.report_warnings import build_warnings


class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str
    use_ai: bool = False


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(value: str | None) -> str:
    return (value or "").strip().lower()


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_RE.match(value))


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


app = FastAPI(title=APP_TITLE, version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    if HOME_PATH.exists():
        return FileResponse(HOME_PATH)
    return JSONResponse({"ok": True, "service": APP_TITLE, "version": VERSION})


@app.get("/dashboard")
def index(request: Request):
    if LOGIN_ENABLED and not is_authed(request):
        return RedirectResponse(url="/login?next=%2Fui&msg=Please%20log%20in%20to%20continue", status_code=302)
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return JSONResponse({"ok": True, "msg": "UI not bundled"})


@app.get("/login", include_in_schema=False)
def login_page():
    if LOGIN_PATH.exists():
        return FileResponse(LOGIN_PATH)
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/register", include_in_schema=False)
def register_page():
    if REGISTER_PATH.exists():
        return FileResponse(REGISTER_PATH)
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/forgot-password", include_in_schema=False)
def forgot_password_page():
    if FORGOT_PASSWORD_PATH.exists():
        return FileResponse(FORGOT_PASSWORD_PATH)
    return RedirectResponse(url="/login", status_code=302)


@app.get("/reset-password", include_in_schema=False)
def reset_password_page():
    if RESET_PASSWORD_PATH.exists():
        return FileResponse(RESET_PASSWORD_PATH)
    return RedirectResponse(url="/login", status_code=302)


@app.post("/auth/register")
def auth_register(email: str | None = Form(None), username: str | None = Form(None), password: str = Form(...)):
    if not LOGIN_ENABLED:
        raise HTTPException(400, "Registration disabled")
    u = normalize_email(email or username)
    p = (password or "").strip()
    if not u or not p:
        raise HTTPException(400, "Email and password are required")
    if not is_valid_email(u):
        raise HTTPException(400, "Please provide a valid email address")
    db = SessionLocal()
    try:
        existing = db.query(User).filter_by(username=u).first()
        if existing:
            raise HTTPException(409, "Email already exists")
        user = User(username=u, password_hash=hash_password(p))
        db.add(user)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/auth/login")
def auth_login(request: Request, email: str | None = Form(None), username: str | None = Form(None), password: str = Form(...)):
    if not LOGIN_ENABLED:
        return {"ok": True, "redirect": "/dashboard"}
    if is_authed(request):
        return {"ok": True, "redirect": "/dashboard", "already_logged_in": True}

    u = normalize_email(email or username)
    p = (password or "").strip()
    if not is_valid_email(u):
        raise HTTPException(400, "Please enter a valid email")
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=u).first()
    finally:
        db.close()

    ok = False
    if user and verify_password(p, user.password_hash):
        ok = True
    if not ok:
        raise HTTPException(401, "Invalid credentials")

    tok = make_token(u)
    resp = JSONResponse({"ok": True, "redirect": "/dashboard"})
    resp.set_cookie(SESSION_COOKIE, tok, httponly=True, samesite="lax", secure=COOKIE_SECURE, max_age=60 * 60 * 24 * 7)
    return resp


@app.post("/auth/forgot-password")
def auth_forgot_password(request: Request, email: str = Form(...)):
    # Intentionally return the same response whether account exists or not.
    generic = {"ok": True, "message": "If this email exists, a reset link was sent."}
    if not LOGIN_ENABLED:
        return generic

    addr = normalize_email(email)
    if not is_valid_email(addr):
        return generic

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=addr).first()
        if not user:
            return generic

        raw = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_TTL_MINUTES)
        rec = PasswordResetToken(email=addr, token_hash=token_hash, expires_at=expires_at, used=False)
        db.add(rec)
        db.commit()
    finally:
        db.close()

    base_url = APP_BASE_URL or str(request.base_url).rstrip("/")
    reset_link = f"{base_url}/reset-password?token={raw}"
    body = (
        "We received a request to reset your password.\n\n"
        f"Reset link: {reset_link}\n\n"
        f"This link expires in {RESET_TOKEN_TTL_MINUTES} minutes."
    )
    sent = send_email(addr, "Reset your ATS account password", body)
    if not sent:
        print("Password reset email was not delivered. Reset link for debugging:", reset_link)

    return generic


@app.post("/auth/reset-password")
def auth_reset_password(token: str = Form(...), password: str = Form(...)):
    raw_token = (token or "").strip()
    new_password = (password or "").strip()
    if len(new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not raw_token:
        raise HTTPException(400, "Invalid reset token")

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    now = datetime.utcnow()

    db = SessionLocal()
    try:
        rec = (
            db.query(PasswordResetToken)
            .filter_by(token_hash=token_hash, used=False)
            .order_by(PasswordResetToken.created_at.desc())
            .first()
        )
        if not rec or rec.expires_at < now:
            raise HTTPException(400, "Reset token is invalid or expired")

        user = db.query(User).filter_by(username=rec.email).first()
        if not user:
            raise HTTPException(400, "Reset token is invalid or expired")

        user.password_hash = hash_password(new_password)
        rec.mark_used()
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/auth/logout")
def auth_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "version": VERSION}


@app.get("/r/{rid}")
def get_report(rid: str):
    p = REPORT_DIR / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(p, media_type="application/json")


@app.get("/report/{rid}.pdf")
def get_report_pdf(rid: str):
    p_json = REPORT_DIR / f"{rid}.json"
    if not p_json.exists():
        raise HTTPException(404, "Report not found")
    p_pdf = REPORT_DIR / f"{rid}.pdf"

    try:
        if (not p_pdf.exists()) or (p_pdf.stat().st_mtime < p_json.stat().st_mtime):
            report = json.loads(p_json.read_text("utf-8"))
            build_pdf_from_report(report, p_pdf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to build PDF: {e}")

    return FileResponse(p_pdf, media_type="application/pdf", filename="ATS-report.pdf")


@app.post("/score-text")
def score_text(request: Request, payload: ScoreRequest):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    rate_limit(request)

    resume_text = sanitize_unicode((payload.resume_text or "").strip())
    jd_text = sanitize_unicode((payload.job_description or "").strip())
    if not jd_text:
        raise HTTPException(400, "job_description is required.")
    validate_jd_hard(jd_text)
    if not resume_text:
        raise HTTPException(400, "resume_text is empty.")

    report = compute_score(resume_text, jd_text)
    report = apply_ai_grading(
        report,
        resume_text,
        jd_text,
        requested=payload.use_ai,
    )
    jd_warnings, jd_quality_score, jd_low_conf = assess_jd_quality(jd_text)
    report["input_quality_score"] = jd_quality_score
    report["low_confidence"] = jd_low_conf
    report["format_warnings"] = build_warnings(report, ext="txt", r_bytes=b"", docx_stats=None) + jd_warnings

    rid = str(uuid.uuid4())
    report["id"] = rid
    report["share_url"] = f"/r/{rid}"
    (REPORT_DIR / f"{rid}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return report


@app.post("/score", include_in_schema=False)
def score_legacy(request: Request, payload: ScoreRequest):
    """
    Backwards-compatible alias used by older front-end bundles.
    """
    return score_text(request, payload)


@app.post("/score-file")
async def score_file(
    request: Request,
    resume: UploadFile = File(...),
    job_description: UploadFile | None = File(None),
    jd_text: str | None = Form(None),
    use_ai: bool = Form(False),
):
    if not is_authed(request):
        raise HTTPException(401, "Login required")
    rate_limit(request)

    r_bytes = await resume.read()
    if len(r_bytes) == 0:
        raise HTTPException(400, "Empty resume file.")
    ensure_size(r_bytes)
    r_name = resume.filename or "resume"
    r_ext = ext_from_filename(r_name)

    if job_description is not None:
        jd_bytes = await job_description.read()
        ensure_size(jd_bytes)
        jd_ext = ext_from_filename(job_description.filename or "jd")
        if jd_ext == "pdf":
            jd_text2 = read_pdf_bytes(jd_bytes)
        elif jd_ext == "docx":
            jd_text2 = read_docx_bytes(jd_bytes)
        else:
            try:
                jd_text2 = jd_bytes.decode("utf-8", errors="ignore")
            except Exception:
                jd_text2 = ""
    else:
        jd_text2 = jd_text or ""

    jd_text2 = sanitize_unicode((jd_text2 or "").strip())
    validate_jd_hard(jd_text2)

    resume_text = ""
    if r_ext == "pdf":
        resume_text = read_pdf_bytes(r_bytes)
    elif r_ext == "docx":
        resume_text = read_docx_bytes(r_bytes)
    else:
        try:
            resume_text = r_bytes.decode("utf-8", errors="ignore")
        except Exception:
            resume_text = ""

    resume_text = sanitize_unicode((resume_text or "").strip())

    if not resume_text.strip():
        raise HTTPException(400, "No text found in resume. If it’s a scanned PDF, convert to DOCX or use OCR.")

    report = compute_score(resume_text, jd_text2)
    report = apply_ai_grading(
        report,
        resume_text,
        jd_text2,
        requested=use_ai,
    )
    jd_warnings, jd_quality_score, jd_low_conf = assess_jd_quality(jd_text2)
    docx_meta = docx_stats_from_bytes(r_bytes) if r_ext == "docx" else None
    report["input_quality_score"] = jd_quality_score
    report["low_confidence"] = jd_low_conf
    report["format_warnings"] = build_warnings(report, ext=r_ext, r_bytes=r_bytes, docx_stats=docx_meta) + jd_warnings

    rid = str(uuid.uuid4())
    report["id"] = rid
    report["share_url"] = f"/r/{rid}"
    (REPORT_DIR / f"{rid}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return report


@app.get("/ads.txt")
def ads_txt():
    if ADS_PATH.exists():
        return FileResponse(ADS_PATH, media_type="text/plain")
    if ADS_FALLBACK:
        return PlainTextResponse(ADS_FALLBACK, media_type="text/plain")
    raise HTTPException(404, "ads.txt file not found")
