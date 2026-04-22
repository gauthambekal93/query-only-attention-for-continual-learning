# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:41:46 2026

@author: gauthambekal93
"""

import os

os.environ["KAGGLEHUB_CACHE"] = r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/datasets/imagenet_1k"

import kagglehub

# Download latest version
path = kagglehub.competition_download('imagenet-object-localization-challenge')

print("Path to competition files:", path)