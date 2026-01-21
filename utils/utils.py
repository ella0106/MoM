import re
import os
import os.path as osp
import json
import cv2
import math
import random
import numpy as np
import pandas as pd
import pickle as pkl
import subprocess
import argparse
import functools
import sys

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, Subset, IterableDataset, get_worker_info
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

from transformers import get_linear_schedule_with_warmup

import deepspeed
import torch.distributed as dist

from tqdm import tqdm
from .constants import *
from decord import VideoReader

def load_file(file_name):
    ext = osp.splitext(file_name)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_name)

    if ext == ".parquet":
        return pd.read_parquet(file_name)

    if ext == ".txt":
        with open(file_name, "r", encoding="utf-8") as fp:
            annos = fp.readlines()
        return [line.rstrip() for line in annos]

    if ext == ".json":
        with open(file_name, "r", encoding="utf-8") as fp:
            return json.load(fp)

    raise ValueError(f"Unsupported file type: {file_name}")

def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = osp.dirname(filename)
    if filepath != '' and not osp.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)
            
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)
        
def fast_tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int,
    modality: str,
):
    # 1) split 기준 결정
    sep = "<video>" if modality == "video" else "<image>"
    chunks = prompt.split(sep)

    # 2) 각 chunk를 tokenize (하지만 결과는 tensor로)
    token_chunks = [
        tokenizer(chunk, add_special_tokens=False).input_ids
        for chunk in chunks
    ]

    # 3) interleave (토큰 + image 토큰)
    # image token은 chunk 사이마다 1개
    out = []
    for i, ids in enumerate(token_chunks):
        out.extend(ids)
        if i < len(token_chunks) - 1:
            out.append(image_token_index)

    return torch.tensor(out, dtype=torch.long)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None, modality=None):
    if modality == "video":
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<video>")]
    else:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

def load_video(video_path, max_frames_num=64,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    # print("video_path:",video_path)
    vr = VideoReader(video_path)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image