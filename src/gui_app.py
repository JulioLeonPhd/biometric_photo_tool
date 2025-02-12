# gui_app.py

import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import ImageTk, Image

import biometric_photo_face_recognition as bp

class BiometricApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Biometric Photo Editor (with Whiten BG)")
        self.geometry("900x600")

        # Variables / State
        self.input_file = None
        self.raw_data = None    # if RAW, store rawpy object
        self.nonraw_bgr = None  # if non-RAW, store the loaded BGR
        self.bbox = None
        
        self.rotation_degs = 0
        self.face_coverage = tk.DoubleVar(value=0.75)
        self.brightness = tk.DoubleVar(value=1.0)
        
        self.show_grid = tk.BooleanVar(value=False)
        self.do_whiten_bg = tk.BooleanVar(value=False)  # NEW: whether to whiten background

        self.create_widgets()

    def create_widgets(self):
        # Left frame for controls
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right frame for preview
        preview_frame = tk.Frame(self)
        preview_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # =========== CONTROL FRAME =============

        tk.Button(control_frame, text="Select Input File", command=self.select_input_file).pack(pady=5, anchor="w")

        tk.Label(control_frame, text="Face Coverage").pack(anchor="w")
        tk.Scale(control_frame,
                 variable=self.face_coverage, from_=0.4, to=0.9, resolution=0.01,
                 orient=tk.HORIZONTAL, length=150).pack(anchor="w")

        tk.Label(control_frame, text="Brightness (RAW only)").pack(anchor="w")
        tk.Scale(control_frame,
                 variable=self.brightness, from_=0.5, to=8.0, resolution=0.1,
                 orient=tk.HORIZONTAL, length=150).pack(anchor="w")

        tk.Button(control_frame, text="Update Preview", command=self.update_preview).pack(pady=5)

        rotate_frame = tk.Frame(control_frame, bd=2, relief=tk.GROOVE)
        rotate_frame.pack(pady=5, fill=tk.X)
        tk.Label(rotate_frame, text="Rotation (1° steps)").pack(anchor="w")
        tk.Button(rotate_frame, text="Rotate -1°", command=lambda: self.rotate_image(-1)).pack(side=tk.LEFT, padx=5)
        tk.Button(rotate_frame, text="Rotate +1°", command=lambda: self.rotate_image(1)).pack(side=tk.LEFT, padx=5)

        shift_frame = tk.Frame(control_frame, bd=2, relief=tk.GROOVE)
        shift_frame.pack(pady=5, fill=tk.X)
        tk.Label(shift_frame, text="Move Crop (Inverted)").pack(anchor="w")
        tk.Button(shift_frame, text="Up", command=lambda: self.shift_bbox(0, 10)).pack()
        tk.Button(shift_frame, text="Down", command=lambda: self.shift_bbox(0, -10)).pack()
        tk.Button(shift_frame, text="Left", command=lambda: self.shift_bbox(10, 0)).pack()
        tk.Button(shift_frame, text="Right", command=lambda: self.shift_bbox(-10, 0)).pack()

        # NEW: Whiten BG checkbox
        tk.Checkbutton(control_frame, text="Whiten BG", variable=self.do_whiten_bg,
                       command=self.update_preview).pack(anchor="w", pady=5)

        # Existing "Show grid" checkbox
        tk.Checkbutton(control_frame, text="Show Grid on Preview",
                       variable=self.show_grid, command=self.update_preview).pack(anchor="w", pady=5)

        # "Export Single Photo 600 DPI" button
        tk.Button(control_frame, text="Export Single Photo (600 DPI)",
                  command=self.export_single_600dpi).pack(pady=5)

        # "Export 3x2 Layout" button
        tk.Button(control_frame, text="Export 3x2 Layout",
                  command=self.export_layout).pack(pady=5)

        # The preview label
        self.preview_label = tk.Label(preview_frame, bg="white", text="[No Preview]")
        self.preview_label.pack(expand=True, fill=tk.BOTH)

    def select_input_file(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp *.cr2 *.nef *.arw *.dng *.raf *.rw2 *.orf *.srw"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select portrait file", filetypes=filetypes)
        if not path:
            return

        self.input_file = path
        self.raw_data = bp.load_raw_data(path)
        self.nonraw_bgr = None

        # If RAW, postprocess once for face detection
        if self.raw_data is not None:
            init_bgr = bp.convert_raw_to_bgr(self.raw_data, bright=self.brightness.get())
        else:
            # Non-RAW
            init_bgr = bp.load_nonraw_bgr(path)

        self.nonraw_bgr = init_bgr

        # Detect face => store bounding box
        face_bbox = bp.detect_face_bounding_box(init_bgr)
        if face_bbox is None:
            messagebox.showwarning("No Face Found", "No face detected. We'll guess a center bounding box.")
            h, w = init_bgr.shape[:2]
            face_bbox = (w//4, h//4, w//2, h//2)
        self.bbox = list(face_bbox)

        self.rotation_degs = 0
        self.update_preview()

    def rotate_image(self, delta_degs):
        self.rotation_degs += delta_degs
        self.update_preview()

    def shift_bbox(self, dx, dy):
        if self.bbox is None:
            return
        self.bbox[0] += dx
        self.bbox[1] += dy
        self.update_preview()

    def update_preview(self):
        """Generate the single 3.5x4.5 preview with rotation, brightness (for RAW),
           optional background whitening, etc."""
        if self.bbox is None:
            self.preview_label.config(image="", text="[No bounding box]")
            return

        # 1) Re-postprocess if RAW with current brightness
        if self.raw_data is not None:
            br = self.brightness.get()
            base_bgr = bp.convert_raw_to_bgr(self.raw_data, bright=br)
        else:
            base_bgr = self.nonraw_bgr

        # 2) Rotate
        rotated_bgr = bp.rotate_image(base_bgr, self.rotation_degs)

        # 3) Optionally whiten background
        if self.do_whiten_bg.get():
            # threshold=200 is a typical guess; you can make it user-adjustable
            rotated_bgr = bp.whiten_background(rotated_bgr, threshold=200)

        # 4) Crop to face coverage
        face_coverage = self.face_coverage.get()
        x, y, w, h = self.bbox
        crop_box = bp.compute_biometric_crop(rotated_bgr, (x, y, w, h), face_coverage)

        # 5) Resize to 3.5x4.5
        final_pil = bp.crop_and_resize_biometric(rotated_bgr, crop_box)

        # 6) Grid?
        if self.show_grid.get():
            final_pil = final_pil.copy()
            final_pil = bp.draw_grid_on_image(final_pil, lines=2, color=(255,0,0), width=2)

        # 7) Downscale for display
        max_preview_width = 400
        ratio = max_preview_width / float(final_pil.width)
        preview_h = int(final_pil.height * ratio)
        preview_img = final_pil.resize((max_preview_width, preview_h), Image.Resampling.LANCZOS)

        # 8) Show
        tk_img = ImageTk.PhotoImage(preview_img)
        self.preview_label.config(image=tk_img, text="")
        self.preview_label.image = tk_img
        
    def export_single_600dpi(self):
        """
        Export the final single biometric photo at 3.5x4.5 cm but 600 DPI,
        suitable for further editing in GIMP.
        """
        if self.bbox is None:
            messagebox.showerror("Error", "No bounding box to export.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save Single Photo (600 DPI)",
            defaultextension=".jpg",
            initialfile="biometric_single_600dpi.jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not out_path:
            return

        try:
            # 1) Re-postprocess if RAW
            if self.raw_data is not None:
                br = self.brightness.get()
                base_bgr = bp.convert_raw_to_bgr(self.raw_data, bright=br)
            else:
                base_bgr = self.nonraw_bgr

            # 2) Rotate
            rotated_bgr = bp.rotate_image(base_bgr, self.rotation_degs)

            # 3) Optional whitening
            if self.do_whiten_bg.get():
                rotated_bgr = bp.whiten_background(rotated_bgr, threshold=200)

            # 4) Crop
            face_coverage = self.face_coverage.get()
            x, y, w, h = self.bbox
            crop_box = bp.compute_biometric_crop(rotated_bgr, (x, y, w, h), face_coverage)

            # 5) Resize with 600 DPI
            single_photo_600 = bp.crop_and_resize_biometric(rotated_bgr, crop_box, dpi=600)

            # 6) Save
            single_photo_600.save(out_path, "JPEG")
            messagebox.showinfo("Success", f"Saved single 600 DPI photo to:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export single:\n{e}")

    def export_layout(self):
        """Export final 3x2 layout with optional whitened BG, brightness, rotation, etc."""
        if self.bbox is None:
            messagebox.showerror("Error", "No bounding box to export.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save 3x2 Layout",
            defaultextension=".jpg",
            initialfile="biometric_6up.jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not out_path:
            return

        try:
            # 1) re-postprocess if RAW
            if self.raw_data is not None:
                br = self.brightness.get()
                base_bgr = bp.convert_raw_to_bgr(self.raw_data, bright=br)
            else:
                base_bgr = self.nonraw_bgr

            # 2) rotate
            rotated_bgr = bp.rotate_image(base_bgr, self.rotation_degs)

            # 3) whiten?
            if self.do_whiten_bg.get():
                rotated_bgr = bp.whiten_background(rotated_bgr, threshold=200)

            # 4) crop
            face_coverage = self.face_coverage.get()
            x, y, w, h = self.bbox
            crop_box = bp.compute_biometric_crop(rotated_bgr, (x, y, w, h), face_coverage)

            single_photo = bp.crop_and_resize_biometric(rotated_bgr, crop_box)

            # 5) create 3x2
            six_pack = bp.create_six_pack(single_photo, dpi=300)
            six_pack.save(out_path, "JPEG")
            messagebox.showinfo("Success", f"Saved 3x2 layout to:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")

def main():
    app = BiometricApp()
    app.mainloop()

if __name__ == "__main__":
    main()
