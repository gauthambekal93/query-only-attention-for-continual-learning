# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:26:02 2026

@author: gauthambekal93
"""

import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import sys 

ROOT = Path(__file__).resolve().parent.parent.parent 
sys.path.insert(0, str(ROOT))
dest = os.path.join(ROOT, "datasets", "imagenet_1k")

class ImageNetCLDataset:

    def __init__(self, data_path, batch_size=256, img_size=128, num_workers=1):

        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        self._build_dataset()
        self._build_class_index()

    # =========================
    # BUILD DATASET
    # =========================
    def _build_dataset(self):

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        self.dataset = datasets.ImageFolder(
            root=os.path.join(self.data_path, 'train'),
            transform=self.transform
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.num_classes = len(self.dataset.classes)

        print(f"Loaded ImageNet with {self.num_classes} classes")

    # =========================
    # PRECOMPUTE CLASS INDICES (IMPORTANT FOR SPEED)
    # =========================
    def _build_class_index(self):

        self.class_to_indices = {}

        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        print("Built class index")

    # =========================
    # SIMPLE BATCH SPLIT (like your tiny imagenet)
    # =========================
    def get_next_batch_split(self):

        img, label = next(iter(self.loader))

        B = img.shape[0]

        rand_idx = torch.randperm(B)

        n_train = int(0.8 * B)
        n_val = int(0.1 * B)

        train_idx = rand_idx[:n_train]
        val_idx = rand_idx[n_train:n_train + n_val]
        test_idx = rand_idx[n_train + n_val:]

        return {
            "train_x": img[train_idx],
            "train_y": label[train_idx],
            "val_x": img[val_idx],
            "val_y": label[val_idx],
            "test_x": img[test_idx],
            "test_y": label[test_idx],
        }

    # =========================
    # TASK SAMPLER (IMPORTANT FOR YOUR CL WORK)
    # =========================
    def sample_task(self, classes_per_task=5, samples_per_class=20):

        chosen_classes = random.sample(range(self.num_classes), classes_per_task)

        task_imgs = []
        task_labels = []

        for c in chosen_classes:

            indices = self.class_to_indices[c]

            chosen_idx = random.sample(indices, samples_per_class)

            for idx in chosen_idx:
                img, label = self.dataset[idx]
                task_imgs.append(img)
                task_labels.append(label)

        task_imgs = torch.stack(task_imgs)
        task_labels = torch.tensor(task_labels)

        return task_imgs, task_labels


# =========================
# USAGE EXAMPLE
# =========================
if __name__ == "__main__":

    data_path = os.path.join(ROOT, "datasets", "imagenet_1k", "imagenet-object-localization-challenge", "ILSVRC", "Data", "CLS-LOC")

    dataset = ImageNetCLDataset(
        data_path=data_path,
        batch_size=256,
        img_size=128,   # 🔥 IMPORTANT
        num_workers=1
    )

    # ===== OPTION 1: Batch split (tiny imagenet style)
    #batch = dataset.get_next_batch_split()

    #print(batch["train_x"].shape)

    # ===== OPTION 2: Task-based CL (RECOMMENDED)
    for task_id in range(5):

        x, y = dataset.sample_task(
            classes_per_task=5,
            samples_per_class=20
        )

        print(f"Task {task_id}: {x.shape}, {y.shape}")