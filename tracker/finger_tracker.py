"""
Finger tracker — segueix la punta de l'índex amb MediaPipe Hands i
captura "strokes" en el mateix format que Google QuickDraw espera:

    drawing = [
        [[x0, x1, ...], [y0, y1, ...]],   # stroke 1
        [[x0, x1, ...], [y0, y1, ...]],   # stroke 2
        ...
    ]

Gestos:
  - "Pinch" (índex + polze junts) → comença / continua un stroke
  - Separar dits → tanca el stroke actual
  - Tecla 'c' → esborra el llenç i reinicia
  - Tecla 'p' → prediu amb el model (si es passa al constructor)
  - Tecla 'q' o ESC → surt
  - Tecla 's' → desa el dibuix com a PNG i imprimeix els strokes

Ús bàsic (només captura):
    python -m tracker.finger_tracker

Integrat amb un model PyTorch:
    from tracker import FingerTracker
    tracker = FingerTracker(model=my_model, classes=my_classes,
                            image_size=64, line_width=5, device="mps")
    tracker.run()
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "Cal instal·lar mediapipe i opencv-python:\n"
        "    pip install mediapipe opencv-python"
    ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strokes_to_quickdraw(strokes):
    """Converteix [[(x,y), (x,y), ...], [...]] al format QuickDraw."""
    out = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        xs = [int(p[0]) for p in stroke]
        ys = [int(p[1]) for p in stroke]
        out.append([xs, ys])
    return out


def render_strokes_pil(strokes_qd, image_size=64, line_width=5,
                       canvas_size=256):
    """Renderitza igual que QuickDrawDataset.draw_strokes_to_image."""
    # Normalitza al canvas_size mantenint l'aspect ratio
    if not strokes_qd:
        return Image.new("L", (image_size, image_size), 0)

    all_x = [x for stroke in strokes_qd for x in stroke[0]]
    all_y = [y for stroke in strokes_qd for y in stroke[1]]
    if not all_x:
        return Image.new("L", (image_size, image_size), 0)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    w = max(max_x - min_x, 1)
    h = max(max_y - min_y, 1)
    scale = (canvas_size - 20) / max(w, h)

    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    for stroke in strokes_qd:
        xs = [(x - min_x) * scale + 10 for x in stroke[0]]
        ys = [(y - min_y) * scale + 10 for y in stroke[1]]
        for i in range(len(xs) - 1):
            draw.line((xs[i], ys[i], xs[i + 1], ys[i + 1]),
                      fill=255, width=line_width)
    return image.resize((image_size, image_size))


# ---------------------------------------------------------------------------
# FingerTracker
# ---------------------------------------------------------------------------
class FingerTracker:
    """Captura strokes a partir de la càmera. Opcionalment fa inferència."""

    PINCH_THRESHOLD = 40   # px entre pinta i polze per considerar pinch
    SMOOTHING_WINDOW = 3   # samples per al filtre EMA

    def __init__(self, camera_index=0, mirror=True,
                 model=None, classes=None, image_size=64, line_width=5,
                 device="cpu"):
        self.cap = cv2.VideoCapture(camera_index)
        self.mirror = mirror
        self.model = model
        self.classes = classes
        self.image_size = image_size
        self.line_width = line_width
        self.device = device

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.draw_utils = mp.solutions.drawing_utils

        self.strokes = []          # [[(x,y), ...], [...]]
        self.current_stroke = []
        self.smooth_buf = deque(maxlen=self.SMOOTHING_WINDOW)
        self.last_pred = ""

    # ---- gesture detection ----
    @staticmethod
    def _pixel(landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)

    def _is_pinch(self, hand_landmarks, w, h):
        idx = hand_landmarks.landmark[8]   # punta índex
        thumb = hand_landmarks.landmark[4] # punta polze
        ix, iy = self._pixel(idx, w, h)
        tx, ty = self._pixel(thumb, w, h)
        dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5
        return dist < self.PINCH_THRESHOLD, (ix, iy)

    # ---- model inference ----
    def _predict(self):
        if self.model is None or self.classes is None:
            return ""
        import torch
        from torchvision import transforms

        qd = strokes_to_quickdraw(self.strokes)
        img = render_strokes_pil(qd, image_size=self.image_size,
                                 line_width=self.line_width)
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        x = tfm(img).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top3 = torch.topk(probs, k=min(3, len(self.classes)))
        return " | ".join(
            f"{self.classes[int(i)]} {float(p)*100:.0f}%"
            for p, i in zip(top3.values, top3.indices)
        )

    # ---- main loop ----
    def run(self):
        print("Pinch (índex+polze) per dibuixar. 'c' clear, 'p' predir, "
              "'s' desar, 'q'/ESC sortir.")
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("No s'ha pogut llegir de la càmera.")
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            pinching = False
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                self.draw_utils.draw_landmarks(
                    frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                pinching, (px, py) = self._is_pinch(lm, w, h)
                if pinching:
                    self.smooth_buf.append((px, py))
                    sx = int(np.mean([p[0] for p in self.smooth_buf]))
                    sy = int(np.mean([p[1] for p in self.smooth_buf]))
                    self.current_stroke.append((sx, sy))
                    cv2.circle(frame, (sx, sy), 8, (0, 255, 0), -1)
                else:
                    if self.current_stroke:
                        self.strokes.append(self.current_stroke)
                        self.current_stroke = []
                        self.smooth_buf.clear()

            # Dibuixa el llenç sobre el frame
            self._overlay_canvas(frame)

            # HUD
            hud = f"strokes: {len(self.strokes)}"
            if self.last_pred:
                hud += f" | pred: {self.last_pred}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Finger Tracker — pinch to draw", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                self.strokes = []
                self.current_stroke = []
                self.last_pred = ""
            elif key == ord("p"):
                self.last_pred = self._predict()
                print("Predicció:", self.last_pred)
            elif key == ord("s"):
                self._save_drawing()

        self.cap.release()
        cv2.destroyAllWindows()

    def _overlay_canvas(self, frame):
        """Pinta els strokes capturats sobre el frame."""
        for stroke in self.strokes + [self.current_stroke]:
            for i in range(len(stroke) - 1):
                cv2.line(frame, stroke[i], stroke[i + 1],
                         (0, 200, 255), 3, cv2.LINE_AA)

    def _save_drawing(self):
        ts = int(time.time())
        out_dir = Path("captures")
        out_dir.mkdir(exist_ok=True)
        qd = strokes_to_quickdraw(self.strokes)
        img = render_strokes_pil(qd, image_size=self.image_size,
                                 line_width=self.line_width)
        path = out_dir / f"drawing_{ts}.png"
        img.save(path)
        # També desem els strokes en JSON per poder reproduir
        import json
        with open(out_dir / f"drawing_{ts}.json", "w") as f:
            json.dump(qd, f)
        print(f"Desat a {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--checkpoint", default=None,
                   help="Ruta al .pth per fer inferència amb 'p'.")
    p.add_argument("--categories_file", default="categories.txt")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--line_width", type=int, default=5)
    p.add_argument("--fc_size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()

    model = None
    classes = None
    device = "cpu"
    if args.checkpoint:
        import torch
        # Importem el model des del train.py
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from train import QuickDrawCNN

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        with open(args.categories_file) as f:
            classes = [l.strip() for l in f if l.strip()]
        model = QuickDrawCNN(
            num_classes=len(classes),
            image_size=args.image_size,
            dropout=args.dropout,
            fc_size=args.fc_size,
        ).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Model carregat: {len(classes)} classes a {device}")

    tracker = FingerTracker(
        camera_index=args.camera,
        mirror=not args.no_mirror,
        model=model,
        classes=classes,
        image_size=args.image_size,
        line_width=args.line_width,
        device=device,
    )
    tracker.run()


if __name__ == "__main__":
    main()
