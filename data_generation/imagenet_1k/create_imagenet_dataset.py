# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:33:23 2026

@author: gauthambekal93
"""

import os
import sys
from pathlib import Path
import random
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent.parent 
sys.path.insert(0, str(ROOT))

src = os.path.join(ROOT, "datasets" ,"imagenet_1k","imagenet-object-localization-challenge", "ILSVRC","Data","CLS-LOC", "train")

dest = os.path.join(ROOT, "datasets", "imagenet_1k")


train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

IMG_SIZE = 224   # change to 128 if you want faster

# create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dest, split), exist_ok=True)

for class_name in os.listdir(src):

    class_path = os.path.join(src, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, imgs in splits.items():

        dst_class_path = os.path.join(dest, split, class_name)
        os.makedirs(dst_class_path, exist_ok=True)

        for img_name in imgs:

            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(dst_class_path, img_name)

            try:
                img = Image.open(src).convert("RGB")

                # 🔥 resize here
                img = img.resize((IMG_SIZE, IMG_SIZE))

                img.save(dst_path, "JPEG", quality=90)

            except Exception as e:
                print(f"Skipping {src_path}: {e}")