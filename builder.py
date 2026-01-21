from transformers import AutoTokenizer
from utils.utils import torch
from model.llava_qwen_mom import LlavaQwenMomForCausalLM
from model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

def get_model(args):   
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.model_type == 'mom':
        base_model = LlavaQwenMomForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False, 
            # quantization_config=bnbconfig,
        )
    elif args.model_type == 'qwen':
        config = LlavaQwenConfig.from_pretrained(
            args.model_path,
            use_motion_tower=True,
            mm_projector_type="mlp2x_gelu",
        )
        base_model = LlavaQwenForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False, 
            # quantization_config=bnbconfig,
    )
    
    model = base_model

    model.requires_grad_(False)
    model.get_motion_tower().requires_grad_(True)
    model.model.mm_projector.requires_grad_(True)
    model.model.fusion_module.requires_grad_(True)
    
    max_length = model.config.max_position_embeddings

    return tokenizer, model, max_length