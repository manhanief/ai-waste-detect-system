"""
gui.py
------
Desktop GUI for the AI Waste Detect System.

Run:  bash run_gui.sh
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

MODEL_PATH      = "custom_cnn.keras"
IMG_SIZE        = 224
DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

# ── Colour palette (eco / modern) ─────────────────────────────────────────────
BG         = "#F0F4F8"   # page background
CARD       = "#FFFFFF"   # card surface
HEADER_BG  = "#1B5E20"   # dark green header / footer
ACCENT     = "#2E7D32"   # primary green
ACCENT_LT  = "#A5D6A7"   # light green (muted text on dark bg)
TEXT_DARK  = "#1A2332"
TEXT_MID   = "#546E7A"
TEXT_LIGHT = "#90A4AE"
BORDER     = "#CFD8DC"
GREEN_OK   = "#2E7D32"   # high confidence
ORANGE_MID = "#F57C00"   # medium confidence
RED_LOW    = "#C62828"   # low confidence

# Per-class bar colours
CLASS_COLORS = {
    "cardboard": "#8D6E63",
    "glass":     "#29B6F6",
    "metal":     "#78909C",
    "paper":     "#FFA726",
    "plastic":   "#AB47BC",
}

# Recycling tips shown after a prediction
TIPS = {
    "cardboard": "Flatten boxes before recycling. Remove all tape and staples first.",
    "glass":     "Rinse bottles and jars clean. Never mix with broken or window glass.",
    "metal":     "Rinse cans before disposal. Both aluminium and steel cans are recyclable.",
    "paper":     "Keep paper dry — wet paper cannot be recycled. Shredded paper goes in a sealed bag.",
    "plastic":   "Check the resin code on the bottom. Rinse containers and remove caps.",
}


class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Waste Detect System")
        self.root.geometry("620x860")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror(
                "Model Not Found",
                f"Could not load '{MODEL_PATH}'.\n"
                f"Run  bash run_train.sh  first to train the model.\n\n{e}")
            root.destroy()
            return

        self.class_names = self._load_class_names()
        self._build_ui()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _load_class_names(self):
        try:
            with open("class_names.json") as f:
                return json.load(f)
        except FileNotFoundError:
            return DEFAULT_CLASSES

    def _card(self, parent, pady=(6, 0)):
        """Return a white rounded-looking card frame."""
        frame = tk.Frame(parent, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", padx=16, pady=pady)
        return frame

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, font=("Helvetica", 10, "bold"),
                 bg=CARD, fg=TEXT_DARK).pack(anchor="w", padx=14, pady=(10, 4))

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_about()
        self._build_categories()
        self._build_upload()
        self._build_preview()
        self._build_result()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HEADER_BG, height=80)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="AI Waste Detect System",
                 font=("Helvetica", 22, "bold"),
                 bg=HEADER_BG, fg="white").pack(pady=(16, 2))
        tk.Label(hdr, text="Convolutional Neural Network  |  5-Class Recyclable Classifier",
                 font=("Helvetica", 8), bg=HEADER_BG, fg=ACCENT_LT).pack()

    def _build_about(self):
        card = self._card(self.root, pady=(14, 0))
        self._section_label(card, "About This System")
        tk.Label(card,
                 text=(
                     "Upload a photo of any waste material and the AI will instantly "
                     "identify whether it is cardboard, glass, metal, paper, or plastic.\n"
                     "Correct sorting helps reduce landfill waste and improves recycling rates."
                 ),
                 font=("Helvetica", 9), bg=CARD, fg=TEXT_MID,
                 wraplength=560, justify="left").pack(anchor="w", padx=14, pady=(0, 10))

    def _build_categories(self):
        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=16, pady=(8, 2))
        tk.Label(row, text="Detectable categories:",
                 font=("Helvetica", 9, "bold"), bg=BG, fg=TEXT_DARK).pack(side="left", padx=(0, 8))
        for name, color in CLASS_COLORS.items():
            tk.Label(row, text=f"  {name}  ",
                     font=("Helvetica", 8, "bold"),
                     bg=color, fg="white",
                     padx=4, pady=3).pack(side="left", padx=3)

    def _build_upload(self):
        card = self._card(self.root)
        self._section_label(card, "Step 1 — Choose an Image")

        row = tk.Frame(card, bg=CARD)
        row.pack(anchor="w", padx=14, pady=(0, 12))

        # Use a Frame+Label button so bg colour renders correctly on macOS
        btn_frame = tk.Frame(row, bg=ACCENT, cursor="hand2")
        btn_frame.pack(side="left")
        btn_label = tk.Label(btn_frame, text="  Browse File  ",
                             font=("Helvetica", 11, "bold"),
                             bg=ACCENT, fg="white", padx=14, pady=7,
                             cursor="hand2")
        btn_label.pack()
        for widget in (btn_frame, btn_label):
            widget.bind("<Button-1>", lambda _: self.select_image())
            widget.bind("<Enter>",    lambda _: btn_frame.config(bg=HEADER_BG) or btn_label.config(bg=HEADER_BG))
            widget.bind("<Leave>",    lambda _: btn_frame.config(bg=ACCENT)    or btn_label.config(bg=ACCENT))

        self.file_label = tk.Label(row, text="No file selected yet",
                                   font=("Helvetica", 9), bg=CARD, fg=TEXT_LIGHT)
        self.file_label.pack(side="left", padx=12)

    def _build_preview(self):
        card = self._card(self.root)
        self._section_label(card, "Step 2 — Image Preview")
        self.image_label = tk.Label(card, bg="#E8ECEF",
                                    text="Your selected image will appear here",
                                    font=("Helvetica", 9), fg=TEXT_LIGHT,
                                    width=74, height=13)
        self.image_label.pack(padx=14, pady=(0, 12))

    def _build_result(self):
        card = self._card(self.root)
        self._section_label(card, "Step 3 — Prediction Result")

        self.result_label = tk.Label(card, text="Waiting for image...",
                                     font=("Helvetica", 16, "bold"),
                                     bg=CARD, fg=TEXT_LIGHT)
        self.result_label.pack(anchor="w", padx=14, pady=(0, 2))

        self.tip_label = tk.Label(card, text="",
                                  font=("Helvetica", 9, "italic"),
                                  bg=CARD, fg=TEXT_MID,
                                  wraplength=560, justify="left")
        self.tip_label.pack(anchor="w", padx=14, pady=(0, 6))

        # Divider
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

        tk.Label(card, text="Confidence breakdown",
                 font=("Helvetica", 9, "bold"), bg=CARD,
                 fg=TEXT_MID).pack(anchor="w", padx=14, pady=(4, 2))

        self.bars_frame = tk.Frame(card, bg=CARD)
        self.bars_frame.pack(fill="x", padx=14, pady=(0, 14))

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=HEADER_BG, height=30)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer,
                 text="CSC583 Project  |  AI Waste Detect System  |  Universiti Teknologi MARA",
                 font=("Helvetica", 8), bg=HEADER_BG, fg=ACCENT_LT).pack(pady=7)

    # ── Actions ──────────────────────────────────────────────────────────────

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select a waste image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.file_label.config(text=path.split("/")[-1], fg=TEXT_DARK)
        self._show_image(path)
        self._classify(path)

    def _show_image(self, path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((560, 200))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img, text="",
                                width=img.width, height=img.height)

    def _classify(self, path):
        img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
        probs = self.model.predict(arr, verbose=0)[0]

        top   = int(np.argmax(probs))
        name  = self.class_names[top]
        conf  = probs[top] * 100

        color = GREEN_OK if conf >= 70 else (ORANGE_MID if conf >= 45 else RED_LOW)
        self.result_label.config(
            text=f"Detected: {name.upper()}   ({conf:.1f}% confidence)",
            fg=color)

        tip = TIPS.get(name, "")
        self.tip_label.config(text=f"Tip: {tip}" if tip else "")

        self._draw_bars(probs)

    def _draw_bars(self, probs):
        for w in self.bars_frame.winfo_children():
            w.destroy()

        for name, p in sorted(zip(self.class_names, probs), key=lambda x: -x[1]):
            color = CLASS_COLORS.get(name, ACCENT)
            row = tk.Frame(self.bars_frame, bg=CARD)
            row.pack(fill="x", pady=3)

            tk.Label(row, text=name.capitalize(), width=10, anchor="w",
                     font=("Helvetica", 9, "bold"), bg=CARD,
                     fg=TEXT_DARK).pack(side="left")

            bar_bg = tk.Frame(row, bg="#E8ECEF", width=310, height=16)
            bar_bg.pack(side="left", padx=6)
            bar_bg.pack_propagate(False)

            fill_w = max(int(310 * p), 2 if p > 0 else 0)
            tk.Frame(bar_bg, bg=color, width=fill_w, height=16).pack(side="left")

            tk.Label(row, text=f"{p*100:5.1f}%", width=7, anchor="e",
                     font=("Helvetica", 9), bg=CARD,
                     fg=TEXT_MID).pack(side="left")


if __name__ == "__main__":
    root = tk.Tk()
    ClassifierApp(root)
    root.mainloop()
