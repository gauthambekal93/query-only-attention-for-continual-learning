# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:28:16 2026

@author: gauthambekal93
"""
import os
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from queue import Queue
import time


import math

def hash_to_unit_float(task_id: int, key: int) -> float:
    # simple deterministic hash → [0,1]
    x = math.sin((task_id + 1) * (key + 1) * 12.9898) * 43758.5453
    return x - math.floor(x)


def hash_to_int(task_id: int, key: int, low: int, high: int) -> int:
    u = hash_to_unit_float(task_id, key)
    return int(low + u * (high - low))


def get_task_params(task_id: int):
    return {
        "crop": hash_to_int(task_id, 0, 2, 9),

        "noise": hash_to_unit_float(task_id, 1) * 0.1,

        "blur": hash_to_int(task_id, 2, 0, 5),

        "brightness": hash_to_unit_float(task_id, 3) * 0.3,

        "contrast": hash_to_unit_float(task_id, 4) * 0.3,

        "cutout": hash_to_int(task_id, 5, 0, 12),

        "color_shift": hash_to_unit_float(task_id, 6) * 0.3,

        "desaturate": hash_to_unit_float(task_id, 7),
    }


def augment_batch(x: torch.Tensor, params: dict) -> torch.Tensor:
    B, C, H, W = x.shape
    device = x.device
    x = x.clone()

    # -------------------
    # 1. flip
    flip_mask = torch.rand(B, device=device) < 0.5
    x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

    # -------------------
    # 2. crop
    pad = params["crop"]
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        off_y = torch.randint(0, 2 * pad + 1, (B,), device=device)
        off_x = torch.randint(0, 2 * pad + 1, (B,), device=device)

        y_grid = torch.arange(32, device=device).view(1, 32, 1)
        x_grid = torch.arange(32, device=device).view(1, 1, 32)

        y_idx = y_grid + off_y.view(B, 1, 1)
        x_idx = x_grid + off_x.view(B, 1, 1)

        y_idx = y_idx.unsqueeze(1).expand(-1, C, -1, -1)
        x_idx = x_idx.unsqueeze(1).expand(-1, C, -1, -1)

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, C, 32, 32)
        c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(B, -1, 32, 32)

        x = x[b_idx, c_idx, y_idx, x_idx]

    # -------------------
    # 3. noise
    if params["noise"] > 0:
        x = x + torch.randn_like(x) * params["noise"]

    # -------------------
    # 4. blur
    if params["blur"] > 0:
        k = params["blur"] * 2 + 1
        pad = k // 2
        x = F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=k,
            stride=1
        )

    # -------------------
    # 5. brightness + contrast
    if params["brightness"] > 0 or params["contrast"] > 0:
        mean = x.mean(dim=(2, 3), keepdim=True)

        b = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * params["brightness"]
        c = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * params["contrast"]

        x = (x - mean) * c + mean
        x = x * b

    # -------------------
    # 6. color shift
    if params["color_shift"] > 0:
        scale = 1.0 + (torch.rand(B, 3, 1, 1, device=device) * 2 - 1) * params["color_shift"]
        x = x * scale

    # -------------------
    # 7. desaturate
    if params["desaturate"] > 0:
        gray = x.mean(dim=1, keepdim=True)
        gray = gray.repeat(1, 3, 1, 1)
        alpha = params["desaturate"]
        x = alpha * gray + (1 - alpha) * x

    # -------------------
    # 8. cutout
    if params["cutout"] > 0:
        hole = params["cutout"]
        H, W = x.shape[2], x.shape[3]

        cy = torch.randint(0, H, (B,), device=device)
        cx = torch.randint(0, W, (B,), device=device)

        half = hole // 2

        y = torch.arange(H, device=device).view(1, H, 1).expand(B, -1, W)
        x_coord = torch.arange(W, device=device).view(1, 1, W).expand(B, H, -1)

        cy = cy.view(B, 1, 1)
        cx = cx.view(B, 1, 1)

        mask = (
            (y >= cy - half) & (y <= cy + half) &
            (x_coord >= cx - half) & (x_coord <= cx + half)
        )

        x = x.masked_fill(mask.unsqueeze(1), 0.0)

    return torch.clamp(x, -3.0, 3.0)