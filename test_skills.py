from ats_skills import ALIAS_RE

text = "Built a computer vision pipeline in PyTorch and deployed with TensorRT on GCP."
hits = [canon for alias,(canon,pat) in ALIAS_RE.items() if pat.search(text.lower())]
print("Detected skills:", sorted(set(hits)))
