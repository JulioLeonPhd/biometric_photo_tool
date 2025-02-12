# biometric_photo_face_recognition.py

import os
import cv2
import rawpy
import numpy as np
from PIL import Image, ImageDraw

# Build the absolute path to the Haar cascade in ../lib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # e.g. ROOT_FOLDER/src
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "..", "lib", "haarcascade_frontalface_default.xml")


def load_raw_data(path):
    """
    Return a rawpy.RawPy object if it's a RAW,
    or None if it's a standard image type.
    """
    ext = os.path.splitext(path)[1].lower()
    raw_exts = [".cr2", ".dng", ".nef", ".arw", ".raf", ".rw2", ".orf", ".srw"]
    if ext in raw_exts:
        return rawpy.imread(path)
    else:
        return None

def convert_raw_to_bgr(raw_data, bright=1.0):
    """
    Convert a rawpy.RawPy object to an OpenCV BGR image
    with the given brightness factor.
    """
    rgb = raw_data.postprocess(
        use_camera_wb=True,
        no_auto_bright=True,
        gamma=(2.2, 2.2),
        bright=bright
    )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def load_nonraw_bgr(path):
    """
    For non-raw images (JPG, PNG...), load via cv2.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot load image from {path}")
    return bgr

def detect_face_bounding_box(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar cascade from: {HAAR_CASCADE_PATH}")

    max_dim = 1600
    h, w = cv_img.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale_factor = max_dim / float(max(h, w))
        small_bgr = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    else:
        small_bgr = cv_img

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
    if len(faces) == 0:
        return None

    (x_small, y_small, w_small, h_small) = max(faces, key=lambda r: r[2]*r[3])
    x = int(x_small / scale_factor)
    y = int(y_small / scale_factor)
    w_face = int(w_small / scale_factor)
    h_face = int(h_small / scale_factor)
    return (x, y, w_face, h_face)

def rotate_image(bgr_img, angle_degrees):
    # same as before
    h, w = bgr_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(bgr_img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return rotated

def compute_biometric_crop(cv_img, bbox, face_coverage=0.75):
    # same logic as before
    (x, y, fw, fh) = bbox
    (img_h, img_w, _) = cv_img.shape
    ratio = 35/45
    crop_h = int(round(fh / face_coverage))
    crop_w = int(round(crop_h * ratio))
    face_cx = x + fw//2
    face_cy = y + fh//2
    x1 = face_cx - crop_w//2
    y1 = face_cy - crop_h//2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    if x1<0: x1=0
    if y1<0: y1=0
    if x2>img_w: x2=img_w
    if y2>img_h: y2=img_h
    return (x1,y1,x2,y2)

def crop_and_resize_biometric(bgr_img, crop_box, dpi=300):
    # same
    x1,y1,x2,y2 = crop_box
    cropped_bgr = bgr_img[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    px_per_cm = dpi / 2.54
    w_px = int(round(3.5 * px_per_cm))
    h_px = int(round(4.5 * px_per_cm))
    out_img = pil_img.resize((w_px, h_px), Image.Resampling.LANCZOS)
    out_img.info["dpi"] = (dpi,dpi)
    return out_img

def draw_grid_on_image(pil_img, lines=2, color=(255,0,0), width=1):
    # same as before
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    for i in range(1, lines+1):
        x = w*(i/(lines+1))
        draw.line([(x,0),(x,h)], fill=color, width=width)
        y = h*(i/(lines+1))
        draw.line([(0,y),(w,y)], fill=color, width=width)
    return pil_img

def create_six_pack(bio_img, dpi=300):
    # same as before
    px_per_cm = dpi / 2.54
    w_10 = int(round(10*px_per_cm))
    h_15 = int(round(15*px_per_cm))
    canvas = Image.new("RGB", (w_10,h_15), "white")
    bw,bh = bio_img.size
    margin_x = (w_10 - 2*bw)//3
    margin_y = (h_15 - 3*bh)//4
    for r in range(3):
        for c in range(2):
            x_off = margin_x + c*(bw+margin_x)
            y_off = margin_y + r*(bh+margin_y)
            canvas.paste(bio_img, (x_off, y_off))

    canvas.info["dpi"] = (dpi,dpi)
    return canvas

def whiten_background(bgr_img, threshold=200):
    """
    For each pixel with a grayscale value >= threshold,
    set that pixel to pure white [255,255,255].
    
    bgr_img: OpenCV BGR image (numpy array).
    threshold: integer in [0..255], e.g. 200.
    returns: a *copy* of the image with whitened background.
    
    If you prefer to modify bgr_img in-place, just remove the copy().
    """
    # Convert to grayscale to find near-white regions
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # Create a boolean mask of "pixels >= threshold"
    mask = (gray >= threshold)
    
    # Make a copy so we don't modify the original directly
    out = bgr_img.copy()
    
    # For those pixels, set to pure white
    out[mask] = [255, 255, 255]
    
    return out