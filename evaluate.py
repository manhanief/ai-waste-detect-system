"""
evaluate.py
-----------
Loads a trained model and evaluates it on the held-out test set:
  - overall test accuracy
  - per-class precision, recall, F1-score (classification report)
  - a confusion matrix figure for the report

Run:  python evaluate.py
(Make sure you've trained the matching MODEL_TYPE first.)
"""

import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_loader import get_datasets

# ----------------------------- CONFIG -----------------------------
MODEL_TYPE = "custom"   # "custom" or "transfer" -> loads {MODEL_TYPE}_cnn.keras
# ------------------------------------------------------------------


def main():
    # Load data (we only need the test set + class names here).
    _, _, test_ds, class_names = get_datasets()

    # Prefer the saved class order if present (keeps labels consistent).
    try:
        with open("class_names.json") as f:
            class_names = json.load(f)
    except FileNotFoundError:
        pass

    # Load the trained model.
    model_path = f"{MODEL_TYPE}_cnn.keras"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded {model_path}")

    # Collect true labels and predictions.
    # test_ds was built with shuffle=False, so order is consistent.
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy.
    acc = np.mean(y_true == y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")

    # Per-class precision / recall / F1.
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    # Confusion matrix figure.
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    plt.title(f"Confusion Matrix ({MODEL_TYPE} model)")
    plt.tight_layout()
    out = f"confusion_matrix_{MODEL_TYPE}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved confusion matrix -> {out}")


if __name__ == "__main__":
    main()
