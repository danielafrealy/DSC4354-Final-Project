"""
Step 2: Face detection and cropping (OpenCV Haar cascade).
Single image: python preprocessing/face_crop.py photo.png
Whole folder: python preprocessing/face_crop.py input_dir output_dir
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(image, margin=0.2, size=224):
    """image: file path or PIL image. Returns a cropped size x size PIL RGB face,
    or None if no face is found."""
    if isinstance(image, str):
        image = Image.open(image)
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
    m = int(margin * max(w, h))
    H, W = rgb.shape[:2]
    crop = rgb[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
    return Image.fromarray(crop).resize((size, size))


if __name__ == "__main__":
    src = sys.argv[1]
    if os.path.isfile(src):
        face = crop_face(src)
        if face is None:
            print("No face found")
        else:
            out = "cropped_" + os.path.basename(src)
            face.save(out)
            print("Saved", out)
    else:
        dst = sys.argv[2]
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                face = crop_face(os.path.join(src, name))
                if face is None:
                    print(f"[skip] no face: {name}")
                else:
                    face.save(os.path.join(dst, name))
        print("Done")
