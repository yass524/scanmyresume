# ATS-like Resume Checker (Starter Kit)

A minimal, ready-to-run **ATS checker** that compares a resume against a job description, finds **keyword gaps**, and returns a **score with recommendations**.

## 🚀 Features
- Keyword coverage based on an editable skill taxonomy
- Synonym expansion (e.g., "JS" → "JavaScript")
- Frequency bonus for repeated mentions (capped)
- Simple ATS-friendly hygiene checks (section presence, sentence length)
- CLI tool **and** FastAPI endpoint

## 🧱 Project Structure
```
ats_core.py          # core logic (edit skills/synonyms here)
cli.py               # command-line interface
app.py               # FastAPI service exposing /score
samples/
  sample_resume.txt
  sample_job_description.txt
requirements.txt
README.md
```

## 🛠️ Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ CLI Usage
```bash
python cli.py --resume samples/sample_resume.txt --jd samples/sample_job_description.txt --out report.json
```

## 🌐 API Usage
```bash
uvicorn app:app --reload --port 8080
# Then POST:
# curl -X POST http://localhost:8080/score -H "Content-Type: application/json" \
#   -d "{\"resume_text\": \"...\", \"job_description\": \"...\"}"
```

## 🧩 Customize the Skill Taxonomy
Open `ats_core.py` and extend `SKILL_TAXONOMY` + `SYNONYMS`. The more complete this is for your niche (e.g., data, web, embedded, automotive), the better your scoring and recommendations.

## 🧠 Roadmap (nice-to-have)
- Parse PDF/DOCX uploads (`pdfminer.six`, `python-docx`)
- Auto-extract JD keywords via TF-IDF / noun-chunking (spaCy / scikit-learn)
- Bullet rewrites with action verbs + outcomes
- Multi-language support
- Simple web UI in React (dropzone + results dashboard)

## 💰 Monetization Ideas
- $5–10 per check (Stripe)
- $15–20/month unlimited checks
- White-label to career coaches or university career centers
