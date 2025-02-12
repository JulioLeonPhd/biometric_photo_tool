import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image
import rawpy

# Update this path to your Haar cascade file (included with OpenCV or downloaded separately).
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "haarcascade_frontalface_default.xml")

def load_image_with_camera_wb(input_path):
    """
    Load an image in BGR format (OpenCV):
      - If RAW (CR2, NEF, DNG...), use rawpy with camera-based WB.
      - Otherwise (JPEG, PNG...), load with cv2 directly.
    """
    ext = os.path.splitext(input_path)[1].lower()
    # Typical raw extensions
    raw_exts = [".cr2", ".dng", ".nef", ".arw", ".raf", ".rw2", ".orf", ".srw"]
    
    if ext in raw_exts:
        # Load RAW using rawpy, applying camera white balance
        with rawpy.imread(input_path) as raw:
            # The key here is use_camera_wb=True to apply the in-camera WB (e.g. "Flash")
            rgb = raw.postprocess(
                use_camera_wb=True,    # apply camera WB
                no_auto_bright=True,
                gamma=(2.2, 2.2),        # typical sRGB-ish gamma
                bright=4  # experiment with 1.2, 1.3, etc.
            )
        # Convert from RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        # Load non-RAW file using OpenCV directly
        bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unable to load image from file: {input_path}")
    
    return bgr

def detect_face_bounding_box(cv_img):
    """
    Use OpenCV Haar cascade to detect the largest face in the image.
    Returns (x, y, w, h) for the largest face, or None if no face is found.
    """
    # 1) Possibly downscale the input if it's very large.
    max_dim = 1600  # adjust as needed
    h, w = cv_img.shape[:2]
    scale_factor = 1.0

    if max(h, w) > max_dim:
        scale_factor = max_dim / float(max(h, w))
        # Resize for detection
        cv_img_small = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_AREA)
    else:
        cv_img_small = cv_img

    # 2) Convert to grayscale
    gray_small = cv2.cvtColor(cv_img_small, cv2.COLOR_BGR2GRAY)

    # 3) Load the Haar cascade
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Could not load Haar cascade. Check HAAR_CASCADE_PATH.")

    # 4) Detect faces (tweak parameters if needed)
    faces = face_cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=3, # default 5
        minSize=(30, 30)  # Might adjust to 30,30 if face is small
    )

    if len(faces) == 0:
        return None

    # 5) If multiple faces, pick the largest
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

    # 6) Scale the bounding box back up to the original image size (if we downscaled)
    (x_small, y_small, w_small, h_small) = largest_face
    x = int(x_small / scale_factor)
    y = int(y_small / scale_factor)
    w_box = int(w_small / scale_factor)
    h_box = int(h_small / scale_factor)

    return (x, y, w_box, h_box)


def compute_german_biometric_crop(cv_img, face_bbox):
    """
    Crop so that the face occupies ~75% of the final 3.5:4.5 region,
    per German guidelines (~70-80%). We keep aspect ratio 35:45 => ~0.777.
    face_bbox = (x, y, w, h) from Haar cascade. 
    We'll interpret w,h as the approximate chin-to-top-of-head region.
    """
    x, y, face_w, face_h = face_bbox
    img_h, img_w, _ = cv_img.shape
    
    # Adjust these if you want to zoom in/out more or less
    FACE_COVERAGE = 0.55     # 0.60 => smaller face, 0.80 => larger face
    ASPECT_RATIO = 35 / 45.0 # ~0.777...

    # Face is 75% of final crop height => crop_height_in_pixels = face_h / 0.75
    crop_h = int(round(face_h / FACE_COVERAGE))
    # Keep the aspect ratio => crop_w = crop_h * (35/45)
    crop_w = int(round(crop_h * ASPECT_RATIO))
    
    face_center_x = x + face_w // 2
    face_center_y = y + face_h // 2
    
    # Position crop so face center is in middle
    crop_x1 = face_center_x - crop_w // 2
    crop_y1 = face_center_y - crop_h // 2
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    
    # Clamp to image boundaries
    if crop_x1 < 0:
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y1 = 0
    if crop_x2 > img_w:
        crop_x2 = img_w
    if crop_y2 > img_h:
        crop_y2 = img_h

    # Optionally refine if clamped region changed the ratio
    # We'll keep it simple. If you want to strictly keep 35:45, add more logic here.
    
    return (crop_x1, crop_y1, crop_x2, crop_y2)

def crop_image_to_pil(cv_img, crop_box):
    """
    Crop the BGR OpenCV image using (x1, y1, x2, y2).
    Convert to PIL in RGB for further steps.
    """
    x1, y1, x2, y2 = crop_box
    cropped_bgr = cv_img[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    return pil_img

def resize_to_biometric(pil_img, dpi=300):
    """
    Resize the cropped image to 3.5×4.5 cm at 300 DPI => ~413×531 px.
    """
    px_per_cm = dpi / 2.54  # ~118.11 px/cm at 300 DPI
    width_px = int(round(3.5 * px_per_cm))  # ~413
    height_px = int(round(4.5 * px_per_cm)) # ~531

    resized = pil_img.resize((width_px, height_px), Image.LANCZOS)
    # Optionally store DPI in metadata
    resized.info["dpi"] = (dpi, dpi)
    return resized

def create_six_pack(bio_img, dpi=300):
    """
    Create a 10×15 cm (4×6") canvas at 300 DPI (1181×1772 px approx),
    and paste 6 copies of the biometric photo in a 2×3 grid.
    """
    px_per_cm = dpi / 2.54
    w_10cm = int(round(10 * px_per_cm))  # ~1181
    h_15cm = int(round(15 * px_per_cm))  # ~1772

    canvas = Image.new("RGB", (w_10cm, h_15cm), "white")
    bio_w, bio_h = bio_img.size

    # Basic margin approach
    margin_x = (w_10cm - 2 * bio_w) // 3
    margin_y = (h_15cm - 3 * bio_h) // 4

    # Paste 6 images (2 columns × 3 rows)
    for row in range(3):
        for col in range(2):
            x_offset = margin_x + col * (bio_w + margin_x)
            y_offset = margin_y + row * (bio_h + margin_y)
            canvas.paste(bio_img, (x_offset, y_offset))

    canvas.info["dpi"] = (dpi, dpi)
    return canvas

def main():
    # Minimal Tkinter file dialog
    root = tk.Tk()
    root.withdraw()

    filetypes = [
        ("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp *.cr2 *.nef *.arw *.dng *.raf"),
        ("All files", "*.*")
    ]
    input_path = filedialog.askopenfilename(
        title="Select a portrait file (RAW or JPEG/PNG)",
        filetypes=filetypes
    )

    if not input_path:
        messagebox.showerror("No File Selected", "You didn't select any file.")
        root.destroy()
        sys.exit(1)

    try:
        # 1) Load image with rawpy camera-based WB if RAW
        cv_img = load_image_with_camera_wb(input_path)

        # 2) Detect face
        face_bbox = detect_face_bounding_box(cv_img)
        if face_bbox is None:
            messagebox.showwarning("No Face Detected", "No face found. Try another photo.")
            root.destroy()
            sys.exit(1)

        # 3) Compute German-specific crop
        crop_box = compute_german_biometric_crop(cv_img, face_bbox)

        # 4) Convert the cropped region to PIL
        cropped_pil = crop_image_to_pil(cv_img, crop_box)

        # 5) Resize to 3.5×4.5 cm at 300 DPI
        biometric_pil = resize_to_biometric(cropped_pil, dpi=300)

        # 6) Create final 10×15 cm layout with 6 repeats
        final_canvas = create_six_pack(biometric_pil, dpi=300)

        # 7) Save
        output_path = os.path.join(os.path.dirname(__file__), "..", "output", "biometric_germany_6up.jpg")
        final_canvas.save(output_path, "JPEG")
        messagebox.showinfo("Success", f"Saved 6-up biometric photo to {output_path}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

    root.destroy()

if __name__ == "__main__":
    main()

# import os
# import sys
# import tkinter as tk
# from tkinter import filedialog, messagebox

# import rawpy
# import cv2
# import numpy as np
# from PIL import Image

# # Update with the path to your Haar cascade file.
# # If it's in the same directory, just put its filename:
# HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "haarcascade_frontalface_default.xml")

# def load_image(input_path):
#     """
#     Load image from either RAW (CR2/DNG/NEF/...) or standard format (JPEG/PNG/etc.).
#     Returns an OpenCV BGR image (numpy array) for face detection.
#     """
#     ext = os.path.splitext(input_path)[1].lower()
#     if ext in [".cr2", ".dng", ".nef", ".arw", ".raf"]:
#         # Load RAW using rawpy
#         cv_img = load_image_rawpy_auto_wb(input_path)
#     else:
#         # Load with OpenCV directly (handles JPG, PNG, etc.)
#         cv_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#         if cv_img is None:
#             raise ValueError(f"Unable to load image from file: {input_path}")
#     return cv_img

# def load_image_rawpy_auto_wb(path):
#     with rawpy.imread(path) as raw:
#         # Option A: Use the WB from the camera metadata
#         # (the setting the camera was actually using – e.g., "flash" if you set it in-camera).
#         # rgb = raw.postprocess(
#         #     use_camera_wb=True,
#         #     no_auto_bright=True,   # turn off auto-brightening if you don’t want that
#         #     gamma=(1,1),          # if you want to do your own gamma later
#         # )
        
#         # Option B: Let rawpy guess an auto WB:
#         rgb = raw.postprocess(
#            use_auto_wb=True,
#            no_auto_bright=True,
#            gamma=(1,1),
#         )
        
#     # Now we have an RGB NumPy array, which we can convert to OpenCV BGR if needed:
#     bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#     return bgr

# def white_balance_using_background(bgr_img, top=0, left=0, width=100, height=100):
#     """
#     Sample a region of the image (top-left corner by default)
#     as the 'white' or 'gray' reference, then scale channels so
#     that region becomes truly neutral.
    
#     bgr_img: OpenCV image in BGR format.
#     (top, left, width, height): region to sample.
#     """
#     # 1. Extract region of interest
#     roi = bgr_img[top:top+height, left:left+width]
#     # 2. Compute average color
#     mean_bgr = roi.mean(axis=(0,1))  # shape: (3,) => [B_mean, G_mean, R_mean]
    
#     # 3. We want to scale so that these means become near 255 (pure white)
#     # or you can target [128,128,128] if you'd rather it be "gray."
#     # We'll pick 255 for a "white" target.
#     scale = 255.0 / mean_bgr  # three scale factors, one per channel
    
#     # 4. Apply scale
#     balanced = bgr_img.astype(np.float32)
#     for c in range(3):
#         balanced[:,:,c] = balanced[:,:,c] * scale[c]
    
#     # 5. Clip to valid range [0..255] and convert back to uint8
#     balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    
#     return balanced

# def detect_face_bounding_box(cv_img):
#     """
#     Use OpenCV Haar cascade to detect the face in the image.
#     Returns (x, y, w, h) for the largest face found (in pixels).
    
#     If no face is found, returns None.
#     """
#     # Convert to grayscale for detection
#     gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

#     # Load the Haar cascade
#     face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
#     if face_cascade.empty():
#         raise IOError("Could not load Haar cascade. Check HAAR_CASCADE_PATH.")

#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray, 
#         scaleFactor=1.1, 
#         minNeighbors=5, 
#         minSize=(50, 50),  # Adjust as needed
#     )

#     if len(faces) == 0:
#         return None

#     # If multiple faces, choose the largest by area
#     largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
#     return largest_face  # (x, y, w, h)

# def compute_biometric_crop(cv_img, face_bbox):
#     """
#     Given the face bounding box, define a cropping region that
#     follows an approximate biometric ratio (3.5cm x 4.5cm).
    
#     We'll add some margins around the face so that:
#       - The face is roughly centered vertically.
#       - There's enough space above the head.
    
#     This is a simplistic approach. Adapt as needed for stricter rules
#     (eye line at certain % from top, etc.).
#     """
#     h, w, _ = cv_img.shape
#     x, y, face_w, face_h = face_bbox

#     # We could define a desired aspect ratio (width:height).
#     # For biometric: 3.5 : 4.5 => ratio = 3.5/4.5 ≈ 0.78
#     desired_ratio = 3.5 / 4.5

#     # Let's put a margin factor around the face bounding box.
#     # For instance, we can make the crop 2.0 times the face bounding box in width/height
#     # so that there's some space above, below, left, right.
#     margin_factor = 2.0  
#     crop_w = int(face_w * margin_factor)
#     crop_h = int(face_h * margin_factor)

#     # We also want to ensure the crop respects the desired aspect ratio
#     # We'll adapt the smaller dimension to match the ratio
#     actual_ratio = crop_w / crop_h
#     if actual_ratio < desired_ratio:
#         # The crop is "too tall" (width is narrower relative to height than desired)
#         # So we reduce height to match the ratio
#         desired_crop_h = int(crop_w / desired_ratio)
#         crop_h = desired_crop_h
#     else:
#         # The crop is "too wide"
#         desired_crop_w = int(crop_h * desired_ratio)
#         crop_w = desired_crop_w

#     # Center this crop around the face center
#     face_center_x = x + face_w // 2
#     face_center_y = y + face_h // 2

#     crop_x1 = face_center_x - crop_w // 2
#     crop_y1 = face_center_y - crop_h // 2
#     crop_x2 = crop_x1 + crop_w
#     crop_y2 = crop_y1 + crop_h

#     # Make sure we don't go outside the image boundaries
#     crop_x1 = max(0, crop_x1)
#     crop_y1 = max(0, crop_y1)
#     crop_x2 = min(w, crop_x2)
#     crop_y2 = min(h, crop_y2)

#     # Adjust if the crop got clipped and now the aspect ratio is off
#     # (In practice, you might want to handle edge cases more gracefully.
#     #  For example, if there's not enough space on one side, we might scale down
#     #  margin_factor or shift the crop. We'll keep it simple for now.)
#     final_crop_w = crop_x2 - crop_x1
#     final_crop_h = crop_y2 - crop_y1
#     # Re-check ratio quickly
#     final_ratio = final_crop_w / final_crop_h
#     # If it's drastically off, you could do further adjustments, or just proceed.

#     return (crop_x1, crop_y1, crop_x2, crop_y2)

# def crop_image(cv_img, crop_box):
#     """
#     Crop the OpenCV (BGR) image using (x1, y1, x2, y2) pixel coordinates.
#     Returns a PIL.Image after the crop (so we can easily use Pillow for resizing, etc.).
#     """
#     x1, y1, x2, y2 = crop_box
#     # Crop in OpenCV
#     cropped_bgr = cv_img[y1:y2, x1:x2]  # shape: (rows, cols, channels)
#     # Convert BGR -> RGB
#     cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
#     # Convert to PIL
#     pil_img = Image.fromarray(cropped_rgb)
#     return pil_img

# def resize_to_biometric(pil_img, dpi=300):
#     """
#     Resize the cropped image to 3.5x4.5 cm at given DPI.
#     => ~413x531 pixels at 300 DPI.
#     """
#     px_per_cm = dpi / 2.54  # ~118.11 px/cm at 300 DPI
#     width_px = int(round(3.5 * px_per_cm))  # ~413
#     height_px = int(round(4.5 * px_per_cm)) # ~531

#     resized = pil_img.resize((width_px, height_px), Image.LANCZOS)
#     resized.info["dpi"] = (dpi, dpi)
#     return resized

# def create_six_pack(bio_img, dpi=300):
#     """
#     Create a 10x15 cm (4"x6") canvas at 300 DPI
#     and paste six copies of the biometric photo in a 2x3 grid.
#     """
#     px_per_cm = dpi / 2.54
#     w_10cm = int(round(10 * px_per_cm))  # ~1181
#     h_15cm = int(round(15 * px_per_cm))  # ~1772

#     canvas = Image.new("RGB", (w_10cm, h_15cm), "white")
#     bio_w, bio_h = bio_img.size

#     # Basic margin approach
#     margin_x = (w_10cm - 2 * bio_w) // 3
#     margin_y = (h_15cm - 3 * bio_h) // 4

#     # Place 6 images (2 columns x 3 rows)
#     for row in range(3):
#         for col in range(2):
#             x_offset = margin_x + col * (bio_w + margin_x)
#             y_offset = margin_y + row * (bio_h + margin_y)
#             canvas.paste(bio_img, (x_offset, y_offset))

#     canvas.info["dpi"] = (dpi, dpi)
#     return canvas

# def compute_german_biometric_crop(cv_img, face_bbox):
#     """
#     Attempt to crop the image so that the detected face occupies ~75%
#     of the final 3.5 x 4.5 cm region (i.e. 70-80% range per German guidelines).
    
#     face_bbox = (x, y, w, h) from Haar cascade. We'll interpret (w, h) 
#     as approximately chin-to-top-of-head. Then:
#       - final aspect ratio must be 35:45 => ~0.777...
#       - face should occupy ~75% of the final height in the cropped region.
    
#     Returns (crop_x1, crop_y1, crop_x2, crop_y2).
#     """

#     # 1) Unpack bounding box
#     x, y, face_w, face_h = face_bbox
#     # 2) Get image size
#     img_h, img_w, _ = cv_img.shape

#     # Constants (tweak as desired)
#     ASPECT_RATIO = 35 / 45  # ~0.777...
#     FACE_COVERAGE = 0.60    # Target: face is 75% of the final height
#                             # (the middle of 70–80% recommended range)

#     # 3) We want the final crop to have face_h occupy ~75% of total crop height.
#     #    => crop_height_in_original_pixels = face_h / FACE_COVERAGE
#     crop_h = int(round(face_h / FACE_COVERAGE))

#     # 4) The crop must maintain 35:45 ratio. => crop_w = crop_h * (35/45)
#     crop_w = int(round(crop_h * ASPECT_RATIO))

#     # 5) Compute the face center
#     face_center_x = x + face_w // 2
#     face_center_y = y + face_h // 2

#     # 6) Crop so that the face center is the center of the final rectangle
#     crop_x1 = face_center_x - crop_w // 2
#     crop_y1 = face_center_y - crop_h // 2
#     crop_x2 = crop_x1 + crop_w
#     crop_y2 = crop_y1 + crop_h

#     # 7) Clamp to image boundaries
#     if crop_x1 < 0:
#         crop_x1 = 0
#     if crop_y1 < 0:
#         crop_y1 = 0
#     if crop_x2 > img_w:
#         crop_x2 = img_w
#     if crop_y2 > img_h:
#         crop_y2 = img_h

#     # Recompute final crop width/height after clamping
#     final_crop_w = crop_x2 - crop_x1
#     final_crop_h = crop_y2 - crop_y1

#     # 8) If we got clipped and the aspect ratio is now off,
#     #    you can either accept it or do additional logic to shrink the crop
#     #    to restore 35:45 exactly. A simplistic approach:
#     final_ratio = final_crop_w / final_crop_h
#     desired_ratio = ASPECT_RATIO
    
#     if abs(final_ratio - desired_ratio) > 0.01:
#         # Attempt to fix ratio by adjusting whichever dimension is bigger
#         if final_ratio > desired_ratio:
#             # Crop is "too wide" => reduce width
#             new_w = int(round(final_crop_h * desired_ratio))
#             delta = final_crop_w - new_w
#             crop_x1 += delta // 2
#             crop_x2 -= delta // 2
#         else:
#             # Crop is "too tall" => reduce height
#             new_h = int(round(final_crop_w / desired_ratio))
#             delta = final_crop_h - new_h
#             crop_y1 += delta // 2
#             crop_y2 -= delta // 2

#         # Re-clamp again
#         if crop_x1 < 0: crop_x1 = 0
#         if crop_y1 < 0: crop_y1 = 0
#         if crop_x2 > img_w: crop_x2 = img_w
#         if crop_y2 > img_h: crop_y2 = img_h

#     return (crop_x1, crop_y1, crop_x2, crop_y2)

# def main():
#     # Tkinter setup for file selection
#     root = tk.Tk()
#     root.withdraw()

#     filetypes = [
#         ("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp *.cr2 *.nef *.arw *.dng *.raf"),
#         ("All files", "*.*")
#     ]
#     input_path = filedialog.askopenfilename(
#         title="Select portrait file",
#         filetypes=filetypes
#     )

#     if not input_path:
#         messagebox.showerror("No File Selected", "You didn't select any file.")
#         root.destroy()
#         sys.exit(1)

#     # try:
#     # 1) Load image into OpenCV
#     cv_img = load_image(input_path)

#     # 2) Detect face
#     face_bbox = detect_face_bounding_box(cv_img)
#     if face_bbox == None:
#         messagebox.showwarning("No Face Detected", 
#                                 "No face found in the image. Try another photo or adjust parameters.")
#         root.destroy()
#         sys.exit(1)
#     else:
#           # 3) Compute biometric-based crop (center face, keep aspect ratio ~3.5:4.5)
#         crop_box = compute_biometric_crop(cv_img, face_bbox)

#         # 4) Crop and convert to PIL
#         cropped_pil = crop_image(cv_img, crop_box)

#         # 5) Resize to 3.5x4.5 cm
#         biometric_pil = resize_to_biometric(cropped_pil, dpi=300)

#         # 6) Create 10x15 cm with 6 copies
#         final_canvas = create_six_pack(biometric_pil, dpi=300)

#         # 7) Save result
#         output_path = os.path.join(os.path.dirname(__file__), "..", "output", "biometric_6up.jpg")
#         final_canvas.save(output_path, "JPEG")
#         messagebox.showinfo("Success", f"Saved 6-up biometric photo to {output_path}")

#         # Instead of compute_biometric_crop(), call compute_german_biometric_crop():
#         crop_box = compute_german_biometric_crop(cv_img, face_bbox)
#         cropped_pil = crop_image(cv_img, crop_box)  # from your existing code
#         resized_pil = resize_to_biometric(cropped_pil, dpi=300)
#         final_canvas = create_six_pack(resized_pil, dpi=300)
#         output_path = os.path.join(os.path.dirname(__file__), "..", "output", "biometric_germany_6up.jpg")
#         final_canvas.save(output_path, "JPEG")

#         messagebox.showinfo("Success", f"Saved 6-up biometric (german) photo to {output_path}")

#         # except Exception as e:
#         #     print(f"Error: An error occurred:\n{e}")
#         #     # messagebox.showerror("Error", f"An error occurred:\n{e}")

#         root.destroy()

# if __name__ == "__main__":
#     main()
