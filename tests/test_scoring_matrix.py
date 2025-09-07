import os, sys, importlib, types
import pytest

def load_core(emb_on: bool = False) -> types.ModuleType:
    os.environ["EMB_ON"] = "1" if emb_on else "0"
    if emb_on:
        os.environ.setdefault("EMB_THRESH", "0.60")
    if "ats_core" in sys.modules:
        del sys.modules["ats_core"]
    return importlib.import_module("ats_core")

@pytest.fixture
def resume_strong():
    return """
PROFESSIONAL SUMMARY
Junior ML Engineer focusing on computer vision and MLOps. Python, PyTorch, OpenCV.

EXPERIENCE
- Built YOLOv8 pipeline; containerized inference with Docker on Linux.
- Implemented models in PyTorch and optimized OpenCV pre/post-processing.

PROJECTS
- FastAPI microservice for real-time inference; ONNX export.

SKILLS
Python, PyTorch, OpenCV, Docker, Linux, FastAPI, Git
"""

@pytest.fixture
def jd_cv_strong():
    return """
Title: Junior Computer Vision / ML Engineer
Responsibilities:
- Implement and optimize computer vision models in Python using PyTorch and OpenCV.
- Package inference services with Docker and deploy on Linux.
Required:
- Hands-on with PyTorch, OpenCV, and Docker.
Preferred:
- AWS, ONNX, FastAPI.
"""

def test_exact_match_high_score(resume_strong, jd_cv_strong):
    core = load_core(False)
    out = core.compute_score(resume_strong, jd_cv_strong)
    assert out["components"]["required_coverage"] == pytest.approx(100.0, abs=0.1)
    assert out["components"]["overall_coverage"] >= 80.0
    assert out["score"] >= 75.0

def test_alias_matching_counts_as_coverage():
    core = load_core(False)
    resume = "Experience in image processing pipelines with Python; shipped models to prod."
    jd = "We need computer vision and Python."
    out = core.compute_score(resume, jd)
    assert "computer vision" in out["matched_keywords"]
    assert out["components"]["required_coverage"] >= 50.0

def test_bullets_and_action_verbs_ratios():
    core = load_core(False)
    resume = """
EXPERIENCE
- Built a data pipeline.
- Implemented Docker-based deployment.
- Optimized inference latency.
"""
    jd = "docker"
    out = core.compute_score(resume, jd)
    assert out["components"]["bullets_ratio_%"] >= 90.0
    assert out["components"]["action_verb_ratio_%"] >= 66.0

def test_stuffing_penalty_triggers():
    core = load_core(False)
    resume = " ".join(["python"] * 60)
    jd = "python"
    out = core.compute_score(resume, jd)
    assert out["components"]["stuffing_penalty_applied"] is True
    assert out["score"] <= 60.0

@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None, reason="semantics off")
def test_semantic_credit_controls():
    core = load_core(True)
    jd = """
Required: PLC, SCADA, instrumentation, VFD/servo commissioning, CMMS, LOTO.
Preferred: OPC UA, Modbus, Profinet / EtherNet/IP, OEE.
"""
    # Intentionally avoid exact 'plc' term to force semantic credit
    resume = """
SUMMARY
Electrical/Controls engineer with experience maintaining automated packaging lines.

EXPERIENCE
- Troubleshot programmable logic controllers and safety relays on high-speed fillers.
- Configured operator screens and plant historian on SCADA platform.
- Set up motor drives and tuned servos; calibrated field transmitters and sensors.
- Led lockout-tagout and root-cause investigations.
"""
    out = core.compute_score(resume, jd)
    assert out["components"]["ai_on"] is True
    assert out["components"]["ai_semantic_matches"] >= 1
