
from pathlib import Path
import face_verification

BASE_DIR = Path(__file__).resolve().parent

halsey1 = BASE_DIR / "images" / "halsey1.jpg"
halsey2 = BASE_DIR / "images" / "halsey2.jpg"

distance, is_match = face_verification.verify_faces(str(halsey1), str(halsey2))

print("Halsey Face Verification")
print(f"Image 1: {halsey1.name}")
print(f"Image 2: {halsey2.name}")
print(f"Euclidean distance: {distance:.4f}")
print(f"Same person: {is_match}")
