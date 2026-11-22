"""
Step 4: Web interface (detection part).
Run: python app.py   then open http://127.0.0.1:7860
The enrollment/verification tab will be added by the verification side.
"""
import gradio as gr
from detection.detector import DeepfakeDetector
from preprocessing.face_crop import crop_face

det = DeepfakeDetector()


def check_image(image):
    if image is None:
        return None, "Upload an image."
    face = crop_face(image)
    r = det.predict(image, crop=True)
    msg = (f"**Prediction: {r['label'].upper()}**\n\n"
           f"- Fake score: {r['fake_score']:.3f}\n"
           f"- Real score: {r['real_score']:.3f}\n"
           f"- Face detected: {r['face_found']}")
    return face, msg


with gr.Blocks(title="Face Verification + Deepfake Detection") as demo:
    gr.Markdown("# Face Verification + Deepfake Detection")
    with gr.Tab("Deepfake Detection"):
        inp = gr.Image(type="pil", label="Upload face image")
        btn = gr.Button("Check")
        face_out = gr.Image(label="Cropped face")
        result = gr.Markdown()
        btn.click(check_image, inputs=inp, outputs=[face_out, result])
    with gr.Tab("Enrollment / Verification"):
        gr.Markdown("*Coming soon: built by the verification side.*")

if __name__ == "__main__":
    demo.launch()
