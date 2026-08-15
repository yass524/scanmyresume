# ScanMyResume

[ScanMyResume](https://scanmyresume.org) is a deployed ATS-style resume checker that compares a resume with a target job description, highlights keyword and structure gaps, and optionally uses Groq-hosted GPT-OSS for qualitative grading and personalized feedback.

**Live application:** [https://scanmyresume.org](https://scanmyresume.org)

> ScanMyResume estimates resume-to-job alignment. It is not affiliated with an employer ATS and cannot predict whether a candidate will be interviewed or hired.

## Features

- Text, PDF, and DOCX resume parsing, with OCR fallback when optional OCR dependencies are available.
- Required and preferred skill extraction using a canonical skill taxonomy and aliases.
- Deterministic coverage, frequency, section-quality, length, and keyword-stuffing checks.
- GPT-OSS analysis through Groq with structured JSON output, enabled by default in the web UI and user-disableable.
- Bounded hybrid scoring: AI can influence the score without fully replacing the reproducible rule-based grade.
- Personalized strengths, improvements, and fact-preserving bullet rewrites.
- Email/password authentication, password resets, request rate limiting, shareable JSON reports, and PDF exports.
- Graceful fallback to rule-based analysis when the AI provider is disabled, unavailable, or rate-limited.

## Project structure

```text
.
├── app.py                         # Vercel/ASGI compatibility entry point
├── scanmyresume/
│   ├── main.py                    # FastAPI routes and application lifecycle
│   ├── config.py                  # Environment and path configuration
│   ├── auth.py                    # Session authentication and rate limiting
│   ├── database.py                # SQLAlchemy models and database connection
│   ├── ats/
│   │   ├── core.py                # Deterministic scoring engine
│   │   ├── skills.py              # Canonical skills and aliases
│   │   ├── sections.py            # Resume section parsing
│   │   └── lexicon.py             # Stopwords, action verbs, and regexes
│   └── services/
│       ├── ai_grading.py          # Groq GPT-OSS structured evaluation
│       ├── files.py               # PDF/DOCX/text extraction
│       ├── jd_quality.py          # Job-description validation
│       ├── pdf_reports.py         # PDF report generation
│       ├── report_warnings.py     # Formatting warnings
│       └── email.py               # SMTP password-reset email
├── web/                           # HTML frontend and ads.txt
├── tests/                         # Unit and API regression tests
├── scripts/                       # CLI, debugging, and smoke-test scripts
├── requirements.txt
├── Procfile
└── docker-compose.yml
```

Runtime artifacts such as `reports/`, `report.json`, `users.db`, `.env`, caches, and virtual environments are intentionally excluded from deployments and version control.

## How scoring works

The deterministic score in `scanmyresume/ats/core.py` combines:

| Component | Weight |
| --- | ---: |
| Required-skill coverage | 50% |
| Overall required/preferred coverage | 20% |
| Required-skill frequency | 15% |
| Overall skill frequency | 5% |
| Core resume sections | 10% |

Length and keyword-stuffing penalties are applied afterward.

When AI analysis is enabled, GPT-OSS evaluates five constrained dimensions:

| AI dimension | Weight |
| --- | ---: |
| Role alignment | 35% |
| Evidence and impact | 25% |
| Skills coverage | 20% |
| Clarity | 10% |
| ATS readability | 10% |

By default, the AI quality score contributes 25% to the hybrid calculation and cannot move the deterministic score by more than 10 points. These controls are configured through `AI_SCORE_WEIGHT` and `AI_MAX_ADJUSTMENT`. If Groq fails, the original deterministic score is returned.

## Local setup

Requirements: Python 3.11+ and, optionally, PostgreSQL.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

Generate a secure local session secret with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Environment variables

Copy `.env.example` to `.env`. Never commit `.env` or expose `API_KEY` in frontend JavaScript.

| Variable | Purpose | Default |
| --- | --- | --- |
| `SESSION_SECRET` | Signs authentication cookies | Ephemeral value if omitted |
| `LOGIN_ENABLED` | Enables registration and login | `1` |
| `DB_URL` | SQLAlchemy database URL | Local SQLite; `/tmp` SQLite on serverless |
| `APP_BASE_URL` | Public URL used in password-reset links | Request base URL |
| `COOKIE_SECURE` | Sends cookies over HTTPS only | Enabled on serverless |
| `MAX_RPM` | Per-IP application request limit | `120` |
| `MAX_UPLOAD_MB` | Maximum uploaded file size | `8` |
| `API_KEY` | Groq API key; `GROQ_API_KEY` is also accepted | None |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-20b` |
| `GROQ_REASONING_EFFORT` | GPT-OSS reasoning level | `low` |
| `AI_GRADING_ON` | Server-wide AI grading switch | `1` |
| `AI_SCORE_WEIGHT` | AI share of the hybrid score, capped at 0.5 | `0.25` |
| `AI_MAX_ADJUSTMENT` | Maximum AI score movement, capped at 20 | `10` |
| `REPORT_DIR` | Generated report directory | `reports/` or `/tmp/reports` |

SMTP variables are documented in `.env.example` and are required only for password-reset email delivery.

## API

Primary endpoints:

- `POST /score-text` and `POST /score`: score pasted text. Include `use_ai: true` to request AI feedback.
- `POST /score-file`: score an uploaded resume and uploaded or pasted job description.
- `GET /r/{id}`: retrieve a saved JSON report.
- `GET /report/{id}.pdf`: generate or download a PDF report.
- `GET /health`: application health and version.

Authentication endpoints are available under `/auth/*` when `LOGIN_ENABLED=1`.

## Tests

```bash
pytest tests -q
```

The Groq tests mock network access. A real API key is not required for the automated suite. Optional legacy SentenceTransformer integration tests are disabled by default; run them with `RUN_EMBEDDING_TESTS=1 pytest tests -q` after downloading the configured embedding model.

The text-only CLI can be run from the repository root:

```bash
python -m scripts.cli --resume resume.txt --jd job-description.txt
```

## Deployment

The project is already deployed at [scanmyresume.org](https://scanmyresume.org). The root `app.py` exports the FastAPI application for Vercel and other ASGI platforms. `Procfile` provides the same `app:app` entry point for platforms that use it.

For Vercel:

1. Configure secrets in **Project Settings → Environment Variables** rather than uploading `.env`.
2. Set `APP_BASE_URL=https://scanmyresume.org` and `COOKIE_SECURE=1`.
3. Configure `API_KEY` and enable Groq Zero Data Retention before processing real resumes.
4. Use a managed PostgreSQL database through `DB_URL` for persistent accounts. Vercel's `/tmp` SQLite database and generated report files are ephemeral and can disappear between function instances or deployments.

## AI and privacy

AI analysis is enabled by default in the web UI but can be turned off before submission. Before a Groq request, the application redacts detected email addresses, phone numbers, and URLs. The remaining resume and job-description content is still transmitted to Groq, so users must be clearly informed and should not submit unnecessary confidential or sensitive information. Direct API clients must continue to send `use_ai: true`; omitting it preserves the rule-based-only behavior.

Model output can be inaccurate or biased. The server validates the response schema, constrains score influence, forbids invented experience or metrics in the prompt, and falls back to deterministic scoring on provider errors. Users should verify all recommendations before changing or submitting a resume.

## License

No open-source license has been declared for this repository. All rights are reserved unless the project owner adds a license.
