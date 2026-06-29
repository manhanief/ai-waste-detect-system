"""
predict.py
----------
Classify a single image from the command line. Handy for quick tests and
for the video demonstration.

Run:  python predict.py path/to/image.jpg
"""

import sys
import json

import numpy as np
import tensorflow as tf

MODEL_PATH = "custom_cnn.keras"   # change to transfer_cnn.keras if you prefer
IMG_SIZE = 224

# Fallback class order (alphabetical, as Keras loads them).
DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]


def load_class_names():
    try:
        with open("class_names.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_CLASSES


def predict(image_path):
    class_names = load_class_names()
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load + resize. The model handles rescaling internally, so feed raw pixels.
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)   # shape (1, 224, 224, 3)

    probs = model.predict(arr, verbose=0)[0]
    top = int(np.argmax(probs))

    print(f"\nPrediction: {class_names[top]}  ({probs[top]*100:.1f}% confident)\n")
    print("All classes:")
    for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:<12} {p*100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
