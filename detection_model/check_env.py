"""Step 1: check the environment and that the detection model loads."""
import sys
import torch, transformers, cv2, PIL, gradio

print("Python      :", sys.version.split()[0])
print("torch       :", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("opencv      :", cv2.__version__)
print("pillow      :", PIL.__version__)
print("gradio      :", gradio.__version__)

from transformers import pipeline
pipe = pipeline("image-classification",
                model="prithivMLmods/Deep-Fake-Detector-v2-Model",
                device=0 if torch.cuda.is_available() else -1)
print("Detection model loaded. Labels:", pipe.model.config.id2label)
print("Environment OK")
