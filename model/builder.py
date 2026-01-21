import torch
import torch.nn as nn
import re
import os
from .siglip_encoder import SigLipVisionTower
from .encoder import GOPPipeline, CrossAttentionFusion

class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f"Unknown projector type: {projector_type}")

def build_vision_tower(**kwargs):
    vision_tower = "google/siglip-so400m-patch14-384"
    return SigLipVisionTower(vision_tower, **kwargs)

def build_motion_tower(config):
    if not getattr(config, "use_motion_tower", False):
        return None
    return GOPPipeline()

def build_fusion_module(config):
    if not getattr(config, "use_motion_tower", False):
        return IdentityMap()
    return CrossAttentionFusion()