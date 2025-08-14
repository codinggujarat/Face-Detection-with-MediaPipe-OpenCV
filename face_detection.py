# face_detection.py
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# =========================
# Utility / Drawing helpers
# =========================
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def draw_bbox(img, box, color=(0, 255, 0), thickness=2):
    x, y, w, h = [int(v) for v in box]
    h_img, w_img = img.shape[:2]
    x1 = clamp(x, 0, w_img - 1)
    y1 = clamp(y, 0, h_img - 1)
    x2 = clamp(x + w, 0, w_img - 1)
    y2 = clamp(y + h, 0, h_img - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def resize_long_side(img, max_long_side=1600):
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / long_side
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def show_fit(img, title="Preview", max_w=1200, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(title, img)

# ===================================
# Detector 1: MediaPipe Face Detection
# ===================================
mp_fd = mp.solutions.face_detection

def detect_faces_mediapipe(bgr_img, far_range=True, min_conf=0.3):
    """Return list of (x, y, w, h, score)."""
    # Keep image relatively large for small/ far faces
    img = resize_long_side(bgr_img, 1600)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model_sel = 1 if far_range else 0
    with mp_fd.FaceDetection(model_selection=model_sel,
                             min_detection_confidence=min_conf) as fd:
        res = fd.process(rgb)

    faces = []
    if res.detections:
        H, W = img.shape[:2]
        for det in res.detections:
            bb = det.location_data.relative_bounding_box
            x = int(bb.xmin * W); y = int(bb.ymin * H)
            w = int(bb.width * W); h = int(bb.height * H)
            score = float(det.score[0]) if det.score else 0.0
            # Filter extremely tiny boxes
            if w >= 14 and h >= 14:
                faces.append((x, y, w, h, score))
    return img, faces  # note: may be resized

# =============================================
# Detector 2: OpenCV DNN (Res10 SSD - Caffe model)
# =============================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
URL_PROTOTXT = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
URL_CAFFE = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def ensure_dnn_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTOTXT):
        urllib.request.urlretrieve(URL_PROTOTXT, PROTOTXT)
    if not os.path.exists(CAFFEMODEL):
        urllib.request.urlretrieve(URL_CAFFE, CAFFEMODEL)

def detect_faces_dnn(bgr_img, conf_thr=0.35):
    """Return list of (x, y, w, h, score)."""
    ensure_dnn_model()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    img = resize_long_side(bgr_img, 1600)  # keep detail for small faces
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        score = float(detections[0, 0, i, 2])
        if score < conf_thr:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x = max(0, x1); y = max(0, y1)
        ww = max(0, x2 - x1); hh = max(0, y2 - y1)
        if ww >= 12 and hh >= 12:
            faces.append((x, y, ww, hh, score))
    return img, faces

# ===========================================
# Detector 3: Haar Cascade (fallback/ extra)
# ===========================================
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar = cv2.CascadeClassifier(HAAR_PATH)

def detect_faces_haar(bgr_img):
    """Return list of (x, y, w, h, score=1.0)."""
    img = resize_long_side(bgr_img, 1600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces_cv = haar.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )
    faces = [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces_cv]
    return img, faces

# ===========================================
# Unified pipeline for IMAGE and VIDEO
# ===========================================
def detect_image(image_path):
    img0 = cv2.imread(image_path)
    if img0 is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    # Try MediaPipe (far-range), then DNN, then Haar
    img_mp, faces_mp = detect_faces_mediapipe(img0, far_range=True, min_conf=0.3)
    if len(faces_mp) > 0:
        print(f"[IMAGE] Detector: MediaPipe (far-range). Faces: {len(faces_mp)}")
        img_disp = img_mp.copy()
        for (x, y, w, h, s) in faces_mp:
            draw_bbox(img_disp, (x, y, w, h))
        show_fit(img_disp, "Face Detection - Image")
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    img_dnn, faces_dnn = detect_faces_dnn(img0, conf_thr=0.35)
    if len(faces_dnn) > 0:
        print(f"[IMAGE] Detector: OpenCV DNN (Res10). Faces: {len(faces_dnn)}")
        img_disp = img_dnn.copy()
        for (x, y, w, h, s) in faces_dnn:
            draw_bbox(img_disp, (x, y, w, h))
        show_fit(img_disp, "Face Detection - Image")
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    img_haar, faces_haar = detect_faces_haar(img0)
    print(f"[IMAGE] Detector: Haar (fallback). Faces: {len(faces_haar)}")
    img_disp = img_haar.copy()
    for (x, y, w, h, s) in faces_haar:
        draw_bbox(img_disp, (x, y, w, h))
    show_fit(img_disp, "Face Detection - Image")
    cv2.waitKey(0); cv2.destroyAllWindows()

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 1 else 25

    # MediaPipe works great on video; keep it primary
    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        prev_count = -1
        print("Playing video with face detection. Press 'q' to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            # modest resize for speed, but keep detail
            frame = resize_long_side(frame, 960)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fd.process(rgb)

            faces = []
            if res.detections:
                H, W = frame.shape[:2]
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * W); y = int(bb.ymin * H)
                    w = int(bb.width * W); h = int(bb.height * H)
                    if w >= 14 and h >= 14:
                        faces.append((x, y, w, h))
                        draw_bbox(frame, (x, y, w, h))

            if len(faces) != prev_count:
                print(f"[VIDEO] Faces detected: {len(faces)}")
                prev_count = len(faces)

            cv2.imshow("MP4 Video Face Detection", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# File pickers (same as before)
# ==========================
if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()

    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
    )
    if not img_path:
        print("No image selected."); raise SystemExit

    vid_path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[("Video files", "*.mp4;*.mov;*.m4v;*.avi;*.mkv")]
    )
    if not vid_path:
        print("No video selected."); raise SystemExit

    # Run image first (with multi-detector fallback), then video
    detect_image(img_path)
    detect_video(vid_path)