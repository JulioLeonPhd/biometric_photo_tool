import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import rawpy
from PIL import Image

def load_image(input_path):
    """
    Load image from either RAW (CR2/DNG) or a common format (JPEG/PNG).
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".cr2", ".dng", ".nef", ".arw", ".raf"]:
        with rawpy.imread(input_path) as raw:
            rgb = raw.postprocess()
        img = Image.fromarray(rgb)
    else:
        img = Image.open(input_path)
    return img

def crop_portrait(img):
    """
    Crop the image by specifying a bounding box in pixels.
    Replace (100, 100, 600, 800) with your own or use face detection.
    """
    bounding_box = (100, 100, 600, 800)  # (left, top, right, bottom)
    cropped = img.crop(bounding_box)
    return cropped

def resize_to_biometric(cropped_img, dpi=300):
    """
    Resize the cropped image to 3.5x4.5 cm at the specified DPI.
    Approximately 3.5 cm = 413 px, 4.5 cm = 531 px at 300 DPI.
    """
    px_per_cm = dpi / 2.54  # ~118.11 px/cm at 300 DPI
    width_px = int(round(3.5 * px_per_cm))
    height_px = int(round(4.5 * px_per_cm))

    resized = cropped_img.resize((width_px, height_px), Image.LANCZOS)
    resized.info["dpi"] = (dpi, dpi)
    return resized

def create_six_pack(bio_img, dpi=300):
    """
    Create a 10x15 cm canvas at 300 DPI, and paste six copies 
    of the biometric photo in a 2x3 grid.
    """
    px_per_cm = dpi / 2.54
    w_10cm = int(round(10 * px_per_cm))  # ~1181
    h_15cm = int(round(15 * px_per_cm))  # ~1772

    canvas = Image.new("RGB", (w_10cm, h_15cm), "white")
    bio_w, bio_h = bio_img.size

    # Simple margin calculation for uniform spacing
    margin_x = (w_10cm - 2 * bio_w) // 3
    margin_y = (h_15cm - 3 * bio_h) // 4

    # Paste six images in a 2x3 layout
    for row in range(3):
        for col in range(2):
            x_offset = margin_x + col * (bio_w + margin_x)
            y_offset = margin_y + row * (bio_h + margin_y)
            canvas.paste(bio_img, (x_offset, y_offset))

    canvas.info["dpi"] = (dpi, dpi)
    return canvas

def main():
    # Create a minimal Tkinter root, hide the main window
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a portrait image
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
        # 1. Load image
        img = load_image(input_path)

        # 2. Crop
        cropped = crop_portrait(img)

        # 3. Resize
        biometric_photo = resize_to_biometric(cropped, dpi=300)

        # 4. Create final 10x15 layout
        final_canvas = create_six_pack(biometric_photo, dpi=300)

        # 5. Save output
        output_path = "biometric_6up.jpg"
        final_canvas.save(output_path, "JPEG")

        messagebox.showinfo("Success", f"Saved 6-up biometric photo to {output_path}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

    root.destroy()

if __name__ == "__main__":
    main()
