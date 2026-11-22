# Face Verification + Deepfake Detection

**Modality:** Face

## Models
| Task | Model | Citation |
|---|---|---|
| Verification | FaceNet (InceptionResnetV1) | Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering," CVPR 2015 |
| Deepfake detection | [Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) (ViT) | prithivMLmods, Hugging Face (2025); Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021 |

## Repo structure (detection side)
```
check_env.py                  # 1. environment check
preprocessing/face_crop.py    # 2. face detection + cropping
data/README.md                # 3. benchmark data sources
detection/evaluate.py         # 3. run detector on benchmark samples
detection/inference_test.py   # original inference test
detection/detector.py         # 4. reusable detector for the enrollment flow
app.py                        # 4. web interface
```

## 1. Environment setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
```
**Dependency issues resolved**
- OpenCV 5.x no longer ships the Haar face-detection cascade (`cv2.data.haarcascades` is empty), so face cropping failed with `!empty() in function 'detectMultiScale'`. Fixed by pinning `opencv-python<5` in `requirements.txt`.
- `device=0` in the Hugging Face pipeline crashes on machines without an NVIDIA GPU. The code now picks GPU or CPU automatically.
- The model weights (343 MB) exceed GitHub's 100 MB file limit, so they download from Hugging Face at runtime and are excluded in `.gitignore`.
- The model's Hugging Face repo also contains `training_args.bin` (a pickle file) and a ~1 GB training checkpoint. Neither is needed for inference, so we skip them.

## 2. Preprocessing
`preprocessing/face_crop.py` detects the largest face with an OpenCV Haar cascade, adds a 20% margin, and resizes it to 224×224 (the model's input size).
```bash
python preprocessing/face_crop.py samples/Real.png
```

## 3. Benchmark data
See `data/README.md`. Then:
```bash
python detection/evaluate.py --crop
```

## 4. Enrollment flow + web interface
- `detection/detector.py`: deepfake check the enrollment flow calls to reject fake photos
- `app.py`: Gradio web interface with a Deepfake Detection tab
```bash
python app.py
```

## Inference test results
```
samples/Real.png -> [{'label': 'Realism', 'score': 0.729}, {'label': 'Deepfake', 'score': 0.271}]
samples/Fake.png -> [{'label': 'Deepfake', 'score': 0.634}, {'label': 'Realism', 'score': 0.366}]
```
Both images were classified correctly.

## Labor Division
| Member | Responsibilities |
|---|---|
| Advika Govindarajan | Deepfake detection: model selection, inference test, face cropping, benchmark fake samples, detector for enrollment, web interface detection tab |
| Daniela Frealy | Face verification: model selection, inference test, consent forms, enrollment data, enrollment flow |
| Shared | Web interface, report |
