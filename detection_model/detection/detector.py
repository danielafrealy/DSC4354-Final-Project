"""
Step 4: Reusable deepfake detector (called by the enrollment flow).
    from detection.detector import DeepfakeDetector
    det = DeepfakeDetector()
    result = det.predict("photo.png", crop=True)
    if result["label"] == "fake": reject enrollment
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from transformers import pipeline
from preprocessing.face_crop import crop_face

MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"


class DeepfakeDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.pipe = pipeline("image-classification", model=MODEL,
                             device=0 if torch.cuda.is_available() else -1)

    def predict(self, image, crop=True):
        """image: file path or PIL image.
        Returns {'label': 'real'/'fake', 'fake_score', 'real_score', 'face_found'}"""
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        face = crop_face(image) if crop else None
        face_found = face is not None
        scores = {r["label"]: r["score"] for r in self.pipe(face if face_found else image)}

        fake = scores.get("Deepfake", 0.0)
        return {
            "label": "fake" if fake >= self.threshold else "real",
            "fake_score": fake,
            "real_score": scores.get("Realism", 0.0),
            "face_found": face_found,
        }
