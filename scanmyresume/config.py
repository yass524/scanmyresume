import os
import secrets
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps local tooling usable before dependencies are installed.
    load_dotenv = None

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
WEB_DIR = PROJECT_ROOT / "web"
HERE = PROJECT_ROOT  # Backwards-compatible alias for local runtime data paths.
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

INDEX_PATH = WEB_DIR / "index.html"
LOGIN_PATH = WEB_DIR / "login.html"
REGISTER_PATH = WEB_DIR / "register.html"
FORGOT_PASSWORD_PATH = WEB_DIR / "forgot_password.html"
RESET_PASSWORD_PATH = WEB_DIR / "reset_password.html"
HOME_PATH = WEB_DIR / "home.html"
ADS_PATH = WEB_DIR / "ads.txt"
ADS_FALLBACK = os.environ.get(
    "ADS_TXT_CONTENT",
    "google.com, pub-8220581150534873, DIRECT, f08c47fec0942fa0",
).strip()

# Detect serverless/container runtimes (Cloud Run, Vercel, Lambda)
IS_SERVERLESS = any(
    os.environ.get(k)
    for k in ("K_SERVICE", "PORT", "VERCEL", "NOW_REGION", "AWS_LAMBDA_FUNCTION_NAME")
)


def _init_report_dir() -> Path:
    """
    Pick a writable reports dir. On serverless, prefer /tmp.
    Fall back to /tmp if configured/local path is not writable.
    """
    configured = os.environ.get("REPORT_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("/tmp/reports") if IS_SERVERLESS else (HERE / "reports"))
    candidates.append(Path("/tmp/reports"))

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            continue
    raise RuntimeError("Unable to initialize a writable REPORT_DIR.")


REPORT_DIR = _init_report_dir()

APP_TITLE = "ATS-like Resume Checker API"
VERSION = "0.3.4"

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "8"))
MAX_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_RPM = int(os.environ.get("MAX_RPM", "120"))

LOGIN_ENABLED = os.environ.get("LOGIN_ENABLED", "1") == "1"
DEMO_USER_RAW = os.environ.get("DEMO_USER")
DEMO_PASS_RAW = os.environ.get("DEMO_PASS")
DEMO_USER = DEMO_USER_RAW.strip().lower() if isinstance(DEMO_USER_RAW, str) else None
DEMO_PASS = DEMO_PASS_RAW.strip() if isinstance(DEMO_PASS_RAW, str) else None
DEMO_DEMO_ENABLED = bool(DEMO_USER and DEMO_PASS)
SESSION_COOKIE = "ats_session"

_COOKIE_SEC_DEFAULT = "1" if IS_SERVERLESS else "0"
COOKIE_SECURE = bool(int(os.environ.get("COOKIE_SECURE", _COOKIE_SEC_DEFAULT)))

# Prefer SESSION_SECRET from env. If missing, generate an ephemeral secret so
# serverless startup does not crash; sessions may reset across cold starts.
_session_secret_env = os.environ.get("SESSION_SECRET")
if not _session_secret_env:
    if IS_SERVERLESS:
        print("Warning: SESSION_SECRET is not set. Using ephemeral secret; set SESSION_SECRET in Vercel env.")
    _session_secret_env = secrets.token_urlsafe(32)
SESSION_SECRET = _session_secret_env

APP_BASE_URL = os.environ.get("APP_BASE_URL", "").rstrip("/")
RESET_TOKEN_TTL_MINUTES = int(os.environ.get("RESET_TOKEN_TTL_MINUTES", "30"))
