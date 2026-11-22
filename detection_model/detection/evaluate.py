"""
Step 3: Run the detector on the benchmark folders and report accuracy.
Usage (from repo root):
    python detection/evaluate.py            (no cropping)
    python detection/evaluate.py --crop     (with face cropping)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from detection.detector import DeepfakeDetector

parser = argparse.ArgumentParser()
parser.add_argument("--crop", action="store_true")
args = parser.parse_args()

det = DeepfakeDetector()
correct = total = 0
for truth, folder in [("real", "data/benchmark/real"), ("fake", "data/benchmark/fake")]:
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        r = det.predict(os.path.join(folder, name), crop=args.crop)
        ok = r["label"] == truth
        correct += ok
        total += 1
        print(f"{truth}/{name:30s} -> {r['label']:4s} (fake={r['fake_score']:.3f}) {'OK' if ok else 'WRONG'}")

if total:
    print(f"\nAccuracy: {correct}/{total} = {correct / total:.1%}")
else:
    print("No images found in data/benchmark/real or data/benchmark/fake")
