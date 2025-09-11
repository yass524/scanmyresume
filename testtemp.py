from ats_sections import split_sections, canonicalize_sections

text = """
MOHAMED SHERIF

PROFESSIONAL EXPERIENCE
- Built an API...
- Deployed on AWS

PROJECTS
- YOLOv8 training...

EDUCATION
BSc, 2022

SKILLS & TOOLS
Python, PyTorch, Docker
"""

sp, st = split_sections(text)
sp = canonicalize_sections(sp)
print("present:", sp)        # expect experience=True, projects=True, education=True, skills=True, summary=False
print("has experience block:", "experience" in st)
