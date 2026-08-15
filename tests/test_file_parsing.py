import os, sys, importlib, io
import pytest

def load_app(login_off=True, max_upload_mb="8"):
    if login_off:
        os.environ["LOGIN_ENABLED"] = "0"
    os.environ["MAX_UPLOAD_MB"] = max_upload_mb
    os.environ["MAX_RPM"] = "120"
    for name in ("app", "scanmyresume.main", "scanmyresume.auth", "scanmyresume.config"):
        if name in sys.modules:
            del sys.modules[name]
    return importlib.import_module("app")

@pytest.mark.skipif(importlib.util.find_spec("reportlab") is None, reason="reportlab not installed")
def make_pdf_bytes(text="Hello PDF"):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 720, text)
    c.showPage(); c.save()
    return buf.getvalue()

@pytest.mark.skipif(importlib.util.find_spec("PIL") is None or importlib.util.find_spec("reportlab") is None, reason="pillow/reportlab not installed")
def make_scanned_pdf_bytes(text="SCANNED"):
    # create an image with text, embed as image-only PDF
    from PIL import Image, ImageDraw
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import tempfile

    img = Image.new("RGB", (600, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 80), text, fill="black")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawImage(tmp.name, 50, 600, width=500, height=150, mask='auto')
    c.showPage(); c.save()
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
    return buf.getvalue()

@pytest.mark.skipif(importlib.util.find_spec("docx") is None, reason="python-docx not installed")
def make_docx_bytes(text="Hello DOCX"):
    import docx
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph(text)
    d.save(buf)
    return buf.getvalue()

def test_score_file_txt(tmp_path):
    app_mod = load_app()
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    r_txt = tmp_path / "r.txt"
    j_txt = tmp_path / "j.txt"
    r_txt.write_text("OpenCV computer vision on Docker", encoding="utf-8")
    j_txt.write_text("computer vision, docker", encoding="utf-8")

    files = {
        "resume": ("r.txt", r_txt.read_bytes(), "text/plain"),
        "job_description": ("j.txt", j_txt.read_bytes(), "text/plain"),
    }
    resp = client.post("/score-file", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["components"]["required_coverage"] >= 50.0

@pytest.mark.skipif(importlib.util.find_spec("docx") is None, reason="python-docx not installed")
def test_score_file_docx(tmp_path):
    app_mod = load_app()
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    resume = make_docx_bytes("Experience with Docker and computer vision")
    jd     = make_docx_bytes("Required: Docker, computer vision")
    files = {
        "resume": ("r.docx", resume, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "job_description": ("j.docx", jd, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    }
    resp = client.post("/score-file", files=files)
    assert resp.status_code == 200
    assert resp.json()["score"] > 50.0

@pytest.mark.skipif(importlib.util.find_spec("reportlab") is None, reason="reportlab not installed")
def test_score_file_pdf(tmp_path):
    app_mod = load_app()
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    resume = make_pdf_bytes("Python PyTorch Docker")
    jd     = make_pdf_bytes("Required: Python, Docker; Preferred: PyTorch")
    files = {
        "resume": ("r.pdf", resume, "application/pdf"),
        "job_description": ("j.pdf", jd, "application/pdf"),
    }
    resp = client.post("/score-file", files=files)
    assert resp.status_code == 200
    assert resp.json()["components"]["required_coverage"] >= 66.0

@pytest.mark.skipif(importlib.util.find_spec("PIL") is None or importlib.util.find_spec("reportlab") is None, reason="pillow/reportlab not installed")
def test_score_file_scanned_pdf(tmp_path):
    app_mod = load_app()
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)

    resume = make_scanned_pdf_bytes("Image Only")
    jd     = b"Python Docker"
    files = {
        "resume": ("r.pdf", resume, "application/pdf"),
        "job_description": ("j.txt", jd, "text/plain"),
    }
    resp = client.post("/score-file", files=files)
    # image-only PDFs should raise 400 "No text found..."
    assert resp.status_code == 400
