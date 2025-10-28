from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from utils.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.utils import rank0_print, torch, nn
from model.encoder import MVResidualModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model.llava_qwen_mom import LlavaQwenMomForCausalLM

def get_model(args):   
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = LlavaQwenMomForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False, 
        # quantization_config=bnbconfig,
    )
    
    model = base_model

    model.requires_grad_(False)
    model.get_motion_tower().requires_grad_(True)
    
    max_length = model.config.max_position_embeddings

    return tokenizer, model, max_length