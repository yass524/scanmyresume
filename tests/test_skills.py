from scanmyresume.ats.skills import ALIAS_RE

def test_representative_skill_aliases_are_detected():
    text = "Built a computer vision pipeline in PyTorch and deployed with TensorRT on GCP."
    hits = {canonical for _, (canonical, pattern) in ALIAS_RE.items() if pattern.search(text.lower())}
    assert {"computer vision", "pytorch", "tensorrt", "gcp"} <= hits
