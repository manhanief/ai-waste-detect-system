"""
split_dataset.py
----------------
Takes the raw TrashNet folders (one folder per class) and splits the images
into a train/ and test/ directory structure that Keras can read directly.

Input  (after unzipping dataset-resized.zip):
    dataset-resized/
        cardboard/   img1.jpg img2.jpg ...
        glass/       ...
        metal/       ...
        paper/       ...
        plastic/     ...
        trash/       ...   <- we ignore this one (only 137 imgs, imbalanced)

Output:
    dataset/
        train/
            cardboard/ ...
            glass/ ...
            ...
        test/
            cardboard/ ...
            ...

Run once before training:  python split_dataset.py
"""

import os
import random
import shutil

# ----------------------------- CONFIG -----------------------------
SOURCE_DIR  = "dataset-resized"   # folder you got after unzipping TrashNet
OUTPUT_DIR  = "dataset"           # where train/ and test/ will be created
CLASSES     = ["cardboard", "glass", "metal", "paper", "plastic"]  # our 5 classes
TRAIN_RATIO = 0.8                 # 80% train / 20% test
SEED        = 42                  # fixed seed -> reproducible split
VALID_EXT   = (".jpg", ".jpeg", ".png", ".bmp")
# ------------------------------------------------------------------


def link_with_retries(src, dst, retries=3):
    last_error = None
    source_path = os.path.abspath(src)
    dest_path = os.path.abspath(dst)

    for attempt in range(retries):
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if os.path.lexists(dest_path):
                os.remove(dest_path)
            os.symlink(source_path, dest_path)
            return True
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                continue

    print(f"Warning: could not link {src} -> {dst}: {last_error}")
    return False


def split_dataset():
    random.seed(SEED)

    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            f"Source folder '{SOURCE_DIR}' not found. "
            "Unzip dataset-resized.zip and check the path."
        )

    # Fresh output directory each run so re-running doesn't pile up duplicates.
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    summary = []

    for cls in CLASSES:
        src_cls = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(src_cls):
            raise FileNotFoundError(f"Class folder missing: {src_cls}")

        # Collect and shuffle image files for this class.
        images = [f for f in os.listdir(src_cls)
                  if f.lower().endswith(VALID_EXT)]
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        # Create destination folders and copy.
        for subset, files in [("train", train_imgs), ("test", test_imgs)]:
            dst_cls = os.path.join(OUTPUT_DIR, subset, cls)
            os.makedirs(dst_cls, exist_ok=True)
            print(f"Copying {len(files)} files for {cls}/{subset}...")
            for fname in files:
                src = os.path.join(src_cls, fname)
                dst = os.path.join(dst_cls, fname)
                link_with_retries(src, dst)

        summary.append((cls, len(train_imgs), len(test_imgs), len(images)))

    # Print a clean summary table (useful to paste into your report).
    print(f"\nSplit complete ({int(TRAIN_RATIO*100)}% train / "
          f"{int((1-TRAIN_RATIO)*100)}% test, seed={SEED})\n")
    print(f"{'Class':<12}{'Train':>8}{'Test':>8}{'Total':>8}")
    print("-" * 36)
    tot_tr = tot_te = tot_all = 0
    for cls, tr, te, all_ in summary:
        print(f"{cls:<12}{tr:>8}{te:>8}{all_:>8}")
        tot_tr += tr; tot_te += te; tot_all += all_
    print("-" * 36)
    print(f"{'TOTAL':<12}{tot_tr:>8}{tot_te:>8}{tot_all:>8}\n")


if __name__ == "__main__":
    split_dataset()
