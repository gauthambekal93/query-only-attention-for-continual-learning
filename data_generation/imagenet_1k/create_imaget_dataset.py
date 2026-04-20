# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:03:21 2026

@author: gauthambekal93
"""

"""
magenet token from hugging face website: https://huggingface.co/settings/tokens
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

"""

'''
from datasets import load_dataset

ds = load_dataset("ILSVRC/imagenet-1k",
                  cache_dir=r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/datasets/imagenet_1k",
                  num_proc=1  
                  )

'''

import os

# force stability
os.environ["HF_DATASETS_DOWNLOAD_NUM_PROC"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_MAX_RETRIES"] = "50"
os.environ["HF_HUB_TIMEOUT"] = "300"

from datasets import load_dataset
from datasets.utils.file_utils import DownloadConfig

download_config = DownloadConfig(
    max_retries=50,
    num_proc=1,
    resume_download=True
)

ds = load_dataset(
    "ILSVRC/imagenet-1k",
    cache_dir=r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/datasets/imagenet_1k",
    split="train",
    download_config=download_config
)

print(ds)


KGAT_365442b5b106733465af3c7684c8e8f8


export KAGGLE_API_TOKEN=KGAT_365442b5b106733465af3c7684c8e8f8


kaggle competitions list