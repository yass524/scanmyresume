# tests/test_ats_core.py
import os, sys, importlib, types
import pytest

def load_core(emb_on: bool = False) -> types.ModuleType:
    """
    Reload ats_core with EMB_ON toggled to ensure tests are deterministic.
    """
    os.environ["EMB_ON"] = "1" if emb_on else "0"
    # Optional: lower the threshold a touch for CI-flakiness
    if emb_on:
        os.environ.setdefault("EMB_THRESH", "0.60")
    if "ats_core" in sys.modules:
        del sys.modules["ats_core"]
    return importlib.import_module("ats_core")


# ---------- Fixtures with representative texts ----------
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
Required Qualifications:
- Hands-on with PyTorch, OpenCV, and Docker.
Preferred:
- AWS, ONNX, FastAPI.
"""

@pytest.fixture
def jd_controls():
    return """
Title: Maintenance Engineer (Controls & Automation)
Responsibilities:
- Diagnose and repair PLC controlled machinery (Siemens S7 / TIA Portal, Allen-Bradley ControlLogix / Studio 5000).
- Support HMI/SCADA systems (WinCC, FactoryTalk, Ignition).
Required:
- PLC, SCADA, instrumentation, VFD/servo commissioning, CMMS, LOTO.
Preferred:
- OPC UA, Modbus, Profinet / EtherNet/IP, OEE improvements.
"""

@pytest.fixture
def resume_controls_semantic():
    # Purposely avoid exact aliases for some terms to exercise semantic credit
    return """
SUMMARY
Electrical/Controls engineer with experience maintaining automated packaging lines.

EXPERIENCE
- Troubleshot programmable logic controllers and safety relays on high-speed fillers.
- Configured operator screens and plant historian on SCADA platform.
- Set up motor drives and tuned servos; calibrated field transmitters and sensors.
- Led lockout-tagout and root-cause investigations.

SKILLS
Safety, drives, servo tuning, programmable logic controllers, maintenance management system
"""


# ---------- Tests ----------

def test_exact_match_high_score(resume_strong, jd_cv_strong):
    core = load_core(emb_on=False)
    out = core.compute_score(resume_strong, jd_cv_strong)
    assert isinstance(out, dict)
    # Required coverage should be perfect
    assert out["components"]["required_coverage"] == pytest.approx(100.0, abs=0.1)
    # Overall should be high (preferred not strictly required)
    assert out["components"]["overall_coverage"] >= 80.0
    # Score should be healthy
    assert out["score"] >= 75.0


def test_alias_matching_counts_as_coverage():
    core = load_core(emb_on=False)
    resume = """
SUMMARY
Experience in image processing pipelines with Python; shipped models to prod.
"""
    jd = "We need computer vision and Python."
    out = core.compute_score(resume, jd)
    # "image processing" is an alias for "computer vision"
    assert "computer vision" in out["matched_keywords"]
    assert out["components"]["required_coverage"] >= 50.0


def test_section_hygiene_is_100_when_all_present():
    core = load_core(emb_on=False)
    resume = """
SUMMARY
Something.

EXPERIENCE
- Did stuff.

PROJECTS
- Built a thing.

SKILLS
Python.

EDUCATION
BSc.
"""
    jd = "Python"
    out = core.compute_score(resume, jd)
    assert out["components"]["section_hygiene"] == 100.0


def test_bullets_and_action_verbs_ratios():
    core = load_core(emb_on=False)
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
    core = load_core(emb_on=False)
    resume = " ".join(["python"] * 60)  # over-repetition
    jd = "python"
    out = core.compute_score(resume, jd)
    assert out["components"]["stuffing_penalty_applied"] is True
    assert out["score"] <= 60.0  # score should drop due to penalty


def test_length_penalty_on_too_short_resume():
    core = load_core(emb_on=False)
    resume = "Python PyTorch Docker."  # very short
    jd = "Python, PyTorch, Docker"
    out = core.compute_score(resume, jd)
    assert out["components"]["length_penalty_applied"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed"
)
def test_semantic_credit_for_close_phrase(resume_controls_semantic, jd_controls):
    # Enable embeddings for this test
    core = load_core(emb_on=True)
    out = core.compute_score(resume_controls_semantic, jd_controls)

    # Embeddings should be on
    assert out["components"]["ai_on"] is True

    # We avoided exact alias for 'plc' by using 'programmable logic controllers' (plural)
    # Depending on model similarity & threshold, we expect at least one semantic credit.
    assert out["components"]["ai_semantic_matches"] >= 1

    # Required coverage should be > 0 thanks to semantic credit
    assert out["components"]["required_coverage"] > 0.0
