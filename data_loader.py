"""
data_loader.py
--------------
Loads the split dataset into tf.data pipelines, and provides the
augmentation + normalization layers used by the CNN.

Folder structure expected (created by split_dataset.py):
    dataset/
        train/  <- split into train + validation here via VAL_SPLIT
        test/   <- held-out, only touched at final evaluation

Usage from your training script:
    from data_loader import get_datasets, data_augmentation, normalization_layer
    train_ds, val_ds, test_ds, class_names = get_datasets()

Run directly to sanity-check the data and save sample figures:
    python data_loader.py
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------
DATA_DIR    = "dataset"
IMG_SIZE    = 224     # 224 = compatible with MobileNet/ResNet later.
                      # Drop to 128 for a lighter custom CNN if training is slow.
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2     # 20% of the train folder is used for validation
SEED        = 123
# ------------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE


def get_datasets(data_dir=DATA_DIR, img_size=IMG_SIZE,
                 batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED):
    """Return (train_ds, val_ds, test_ds, class_names).

    Images are returned with raw pixel values in [0, 255]; normalization is
    applied as the first layer of the model (see normalization_layer below).
    """
    train_dir = f"{data_dir}/train"
    test_dir  = f"{data_dir}/test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,                      # same seed -> no overlap with train
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,                  # keep order fixed for the confusion matrix
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    # cache() keeps images in memory after first epoch; prefetch() overlaps
    # data loading with training so the GPU/CPU isn't left waiting.
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# Augmentation: applied ONLY at training time (Keras disables these layers
# automatically during inference). Helps the model generalise from ~400
# images/class. Mention these techniques in the report's preprocessing section.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

# Normalization: scales pixels from [0,255] to [0,1]. Use as the first layer
# of a CUSTOM CNN. (If you switch to MobileNet/ResNet, drop this and use that
# model's own preprocess_input instead.)
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)


def _save_grid(ds, class_names, fname, title, augment=False):
    """Save a 3x3 grid of sample images (optionally augmented) for the report."""
    plt.figure(figsize=(8, 8))
    for images, labels in ds.take(1):
        batch = data_augmentation(images) if augment else images
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_datasets()

    # Count batches per split for a quick sanity check.
    print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Val batches:   {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches:  {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Figures you can drop straight into the report.
    _save_grid(train_ds, class_names, "samples_original.png",
               "Sample training images")
    _save_grid(train_ds, class_names, "samples_augmented.png",
               "After data augmentation", augment=True)
