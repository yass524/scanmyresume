from ats_sections import split_sections

sample_resume = """Summary
Machine Learning Engineer with 4+ years experience in AI and Computer Vision.

Experience
Acme AI — ML Engineer
• Built YOLOv8 pipeline with PyTorch
• Deployed TensorRT models on GCP

Projects
License Plate Recognition
• OCR pipeline with Transformers, reached 95% accuracy

Education
B.Sc. in Computer Engineering, GIU 2020

Skills
Python, PyTorch, TensorRT, Docker, AWS
"""

present, blocks = split_sections(sample_resume)

print("=== Sections Present ===")
for k, v in present.items():
    print(f"{k}: {v}")

print("\n=== Section Texts ===")
for k, v in blocks.items():
    print(f"\n--- {k.upper()} ---\n{v}")
