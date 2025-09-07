# cli.py
import argparse, json, os, sys
from ats_core import compute_score

def read_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    # Simple approach: treat all as text; for PDF/DOCX you can add extraction later
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main():
    p = argparse.ArgumentParser(description="ATS-like Resume Checker (MVP)")
    p.add_argument("--resume", required=True, help="Path to resume text file (use plain .txt for MVP)")
    p.add_argument("--jd", required=True, help="Path to job description text file (.txt)")
    p.add_argument("--out", default="ats_report.json", help="Output report JSON path")
    args = p.parse_args()

    resume_text = read_text(args.resume)
    jd_text     = read_text(args.jd)

    report = compute_score(resume_text, jd_text)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report → {args.out}")

if __name__ == "__main__":
    main()
