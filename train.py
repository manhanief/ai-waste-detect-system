"""
train.py
--------
Trains the CNN, evaluates it on the held-out test set, saves the trained
model, and produces the accuracy/loss curves for the report.

Run:  python train.py

Switch MODEL_TYPE between "custom" and "transfer" to run the two experiments.
"""

import json

import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import get_datasets
from model import build_custom_cnn, build_transfer_model

# ----------------------------- CONFIG -----------------------------
MODEL_TYPE    = "custom"   # "custom" or "transfer"
EPOCHS        = 30
LEARNING_RATE = 1e-3
# ------------------------------------------------------------------


def plot_history(history, fname):
    """Save side-by-side accuracy and loss curves (train vs validation)."""
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_r = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_r, acc, label="train")
    plt.plot(epochs_r, val_acc, label="validation")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_r, loss, label="train")
    plt.plot(epochs_r, val_loss, label="validation")
    plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved training curves -> {fname}")


def main():
    # 1. Load data
    train_ds, val_ds, test_ds, class_names = get_datasets()
    num_classes = len(class_names)

    # Save the class order so the GUI / evaluation scripts label predictions
    # correctly (image_dataset_from_directory sorts classes alphabetically).
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # 2. Build model
    if MODEL_TYPE == "custom":
        model = build_custom_cnn(num_classes)
    elif MODEL_TYPE == "transfer":
        model = build_transfer_model(num_classes)
    else:
        raise ValueError("MODEL_TYPE must be 'custom' or 'transfer'")

    # 3. Compile
    #    sparse_categorical_crossentropy because labels are integers (0..4),
    #    not one-hot vectors.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 4. Callbacks
    callbacks = [
        # Stop when validation accuracy stops improving, keep the best weights.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True),
        # Lower the learning rate when validation loss plateaus.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        # Save the best model during training.
        tf.keras.callbacks.ModelCheckpoint(
            f"best_{MODEL_TYPE}.keras", monitor="val_accuracy",
            save_best_only=True),
    ]

    # 5. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # 6. Final evaluation on the held-out test set (never seen during training)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n=== {MODEL_TYPE} model | Test accuracy: {test_acc:.4f} "
          f"| Test loss: {test_loss:.4f} ===")

    # 7. Save final model + curves
    model.save(f"{MODEL_TYPE}_cnn.keras")
    print(f"Saved model -> {MODEL_TYPE}_cnn.keras")
    plot_history(history, f"curves_{MODEL_TYPE}.png")


if __name__ == "__main__":
    main()
