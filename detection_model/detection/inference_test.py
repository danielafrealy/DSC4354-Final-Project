"""
Deepfake detection inference test
Model: prithivMLmods/Deep-Fake-Detector-v2-Model (Vision Transformer)
Source: https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
Run from repo root: python detection/inference_test.py
"""
from transformers import pipeline

# Load the model
# device=0 uses GPU; change to device=-1 to run on CPU
pipe = pipeline('image-classification',
                model="prithivMLmods/Deep-Fake-Detector-v2-Model",
                device=0)

# Predict on test images
for img in ["samples/Real.png", "samples/Fake.png"]:
    result = pipe(img)
    print(img, "->", result)
