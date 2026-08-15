from scanmyresume.ats.sections import split_sections

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

def test_standard_resume_sections_are_detected():
    present, blocks = split_sections(sample_resume)
    assert present["experience"] is True
    assert present["projects"] is True
    assert present["education"] is True
    assert present["skills"] is True
    assert "TensorRT" in blocks["experience"]
