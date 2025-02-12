import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import rawpy
import cv2
import numpy as np
from PIL import Image

# Update with the path to your Haar cascade file.
# If it's in the same directory, just put its filename:
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "haarcascade_frontalface_default.xml")

def load_image(input_path):
    """
    Load image from either RAW (CR2/DNG/NEF/...) or standard format (JPEG/PNG/etc.).
    Returns an OpenCV BGR image (numpy array) for face detection.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".cr2", ".dng", ".nef", ".arw", ".raf"]:
        # Load RAW using rawpy
        with rawpy.imread(input_path) as raw:
            rgb = raw.postprocess()
        # Convert from RGB to BGR for OpenCV
        cv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        # Load with OpenCV directly (handles JPG, PNG, etc.)
        cv_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError(f"Unable to load image from file: {input_path}")
    return cv_img

def detect_face_bounding_box(cv_img):
    """
    Use OpenCV Haar cascade to detect the face in the image.
    Returns (x, y, w, h) for the largest face found (in pixels).
    
    If no face is found, returns None.
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Could not load Haar cascade. Check HAAR_CASCADE_PATH.")

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50),  # Adjust as needed
    )

    if len(faces) == 0:
        return None

    # If multiple faces, choose the largest by area
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    return largest_face  # (x, y, w, h)

def compute_biometric_crop(cv_img, face_bbox):
    """
    Given the face bounding box, define a cropping region that
    follows an approximate biometric ratio (3.5cm x 4.5cm).
    
    We'll add some margins around the face so that:
      - The face is roughly centered vertically.
      - There's enough space above the head.
    
    This is a simplistic approach. Adapt as needed for stricter rules
    (eye line at certain % from top, etc.).
    """
    h, w, _ = cv_img.shape
    x, y, face_w, face_h = face_bbox

    # We could define a desired aspect ratio (width:height).
    # For biometric: 3.5 : 4.5 => ratio = 3.5/4.5 ≈ 0.78
    desired_ratio = 3.5 / 4.5

    # Let's put a margin factor around the face bounding box.
    # For instance, we can make the crop 2.0 times the face bounding box in width/height
    # so that there's some space above, below, left, right.
    margin_factor = 2.0  
    crop_w = int(face_w * margin_factor)
    crop_h = int(face_h * margin_factor)

    # We also want to ensure the crop respects the desired aspect ratio
    # We'll adapt the smaller dimension to match the ratio
    actual_ratio = crop_w / crop_h
    if actual_ratio < desired_ratio:
        # The crop is "too tall" (width is narrower relative to height than desired)
        # So we reduce height to match the ratio
        desired_crop_h = int(crop_w / desired_ratio)
        crop_h = desired_crop_h
    else:
        # The crop is "too wide"
        desired_crop_w = int(crop_h * desired_ratio)
        crop_w = desired_crop_w

    # Center this crop around the face center
    face_center_x = x + face_w // 2
    face_center_y = y + face_h // 2

    crop_x1 = face_center_x - crop_w // 2
    crop_y1 = face_center_y - crop_h // 2
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # Make sure we don't go outside the image boundaries
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)

    # Adjust if the crop got clipped and now the aspect ratio is off
    # (In practice, you might want to handle edge cases more gracefully.
    #  For example, if there's not enough space on one side, we might scale down
    #  margin_factor or shift the crop. We'll keep it simple for now.)
    final_crop_w = crop_x2 - crop_x1
    final_crop_h = crop_y2 - crop_y1
    # Re-check ratio quickly
    final_ratio = final_crop_w / final_crop_h
    # If it's drastically off, you could do further adjustments, or just proceed.

    return (crop_x1, crop_y1, crop_x2, crop_y2)

def crop_image(cv_img, crop_box):
    """
    Crop the OpenCV (BGR) image using (x1, y1, x2, y2) pixel coordinates.
    Returns a PIL.Image after the crop (so we can easily use Pillow for resizing, etc.).
    """
    x1, y1, x2, y2 = crop_box
    # Crop in OpenCV
    cropped_bgr = cv_img[y1:y2, x1:x2]  # shape: (rows, cols, channels)
    # Convert BGR -> RGB
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    # Convert to PIL
    pil_img = Image.fromarray(cropped_rgb)
    return pil_img

def resize_to_biometric(pil_img, dpi=300):
    """
    Resize the cropped image to 3.5×4.5 cm at given DPI.
    => ~413×531 pixels at 300 DPI.
    """
    px_per_cm = dpi / 2.54  # ~118.11 px/cm at 300 DPI
    width_px = int(round(3.5 * px_per_cm))  # ~413
    height_px = int(round(4.5 * px_per_cm)) # ~531

    resized = pil_img.resize((width_px, height_px), Image.LANCZOS)
    resized.info["dpi"] = (dpi, dpi)
    return resized

def create_six_pack(bio_img, dpi=300):
    """
    Create a 10×15 cm (4"×6") canvas at 300 DPI
    and paste six copies of the biometric photo in a 2×3 grid.
    """
    px_per_cm = dpi / 2.54
    w_10cm = int(round(10 * px_per_cm))  # ~1181
    h_15cm = int(round(15 * px_per_cm))  # ~1772

    canvas = Image.new("RGB", (w_10cm, h_15cm), "white")
    bio_w, bio_h = bio_img.size

    # Basic margin approach
    margin_x = (w_10cm - 2 * bio_w) // 3
    margin_y = (h_15cm - 3 * bio_h) // 4

    # Place 6 images (2 columns x 3 rows)
    for row in range(3):
        for col in range(2):
            x_offset = margin_x + col * (bio_w + margin_x)
            y_offset = margin_y + row * (bio_h + margin_y)
            canvas.paste(bio_img, (x_offset, y_offset))

    canvas.info["dpi"] = (dpi, dpi)
    return canvas

def main():
    # Tkinter setup for file selection
    root = tk.Tk()
    root.withdraw()

    filetypes = [
        ("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp *.cr2 *.nef *.arw *.dng *.raf"),
        ("All files", "*.*")
    ]
    input_path = filedialog.askopenfilename(
        title="Select portrait file",
        filetypes=filetypes
    )

    if not input_path:
        messagebox.showerror("No File Selected", "You didn't select any file.")
        root.destroy()
        sys.exit(1)

    try:
        # 1) Load image into OpenCV
        cv_img = load_image(input_path)

        # 2) Detect face
        face_bbox = detect_face_bounding_box(cv_img)
        if not face_bbox:
            messagebox.showwarning("No Face Detected", 
                                   "No face found in the image. Try another photo or adjust parameters.")
            root.destroy()
            sys.exit(1)

        # 3) Compute biometric-based crop (center face, keep aspect ratio ~3.5:4.5)
        crop_box = compute_biometric_crop(cv_img, face_bbox)

        # 4) Crop and convert to PIL
        cropped_pil = crop_image(cv_img, crop_box)

        # 5) Resize to 3.5×4.5 cm
        biometric_pil = resize_to_biometric(cropped_pil, dpi=300)

        # 6) Create 10×15 cm with 6 copies
        final_canvas = create_six_pack(biometric_pil, dpi=300)

        # 7) Save result
        output_path = os.path.join(os.path.dirname(__file__), "..", "output", "biometric_6up.jpg")
        final_canvas.save(output_path, "JPEG")

        messagebox.showinfo("Success", f"Saved 6-up biometric photo to {output_path}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

    root.destroy()

if __name__ == "__main__":
    main()
