from utils.utils import *
from transformers import AutoTokenizer, GenerationConfig
from dataset import CustomDataset, DataCollatorForCustomDataset
from model.llava_qwen_mom import LlavaQwenMomForCausalLM
from model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig

def infer(args):
    rank0_print("Loading model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.model_type == 'mom':
        model = LlavaQwenMomForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False, 
            # quantization_config=bnbconfig,
        )
        processor = model.get_vision_tower().image_processor
    elif args.model_type == 'qwen':
        config = LlavaQwenConfig.from_pretrained(
            args.model_path,
            use_motion_tower=True,
            torch_dtype=torch.float16,
        )
        model = LlavaQwenForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False, 
            # quantization_config=bnbconfig,
        )
        for n, p in model.named_parameters():
            if p.dtype != torch.float16:
                print(n, p.dtype)
        processor = model.get_vision_tower().image_processor
        
    max_length = model.config.max_position_embeddings
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank0_print(f"{device} is available.")
    model.to(device)
    
    rank0_print(f"Model loaded: {args.model_path}")
    rank0_print(f"Max length: {max_length}")
    
    dataset = CustomDataset(
        video_dir=args.video_dir,
        txt=args.dataset,
        temp_dir=args.temp_dir,
        tokenizer=tokenizer,
        conv_template="qwen_1_5",
        train=False,
        gop_num=model.config.gop_num if hasattr(model.config, 'gop_num') else 8,
        fps=model.config.fps if hasattr(model.config, 'fps') else 4,
        image_processor=processor,
        config=model.config,
    )
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = DataCollatorForCustomDataset(pad_token_id=pad_id, ignore_index=IGNORE_INDEX, train=False)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator
    )
    
    gen_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        temperature=1,
        pad_token_id=pad_id
    )
    
    results = {}
    
    skip = args.skip
    skip_len = skip + args.skip_len
    if args.skip_len == 0:
        skip_len = len(dataset)
    subset = Subset(loader.dataset, range(skip, skip_len))
    loader = DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
    )

    for i, batch in enumerate(tqdm(loader, desc=f"Running inference")):
        i += skip
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
              
        cont = model.generate(
            inputs=batch["input_ids"],
            images=batch["images"],
            motion_feats=batch["motion_feats"],
            residual_feats=batch["residual_feats"],
            modalities=batch["modalities"],
            generation_config=gen_config,
        )
        output = tokenizer.decode(cont[0], skip_special_tokens=True)
        results.update({i : output})
        if (i+1) % 10 == 0:
            save_file([results], args.result_path)
    
    save_file([results], args.result_path)
    rank0_print(f"Inference completed ✅ Results saved to: {args.result_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', required=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='pretrained model path')
    parser.add_argument('--model_type', type=str, required=True, help='pretrained model type')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='tokenizer path')
    parser.add_argument('--video_dir', type=str, required=True, help='video directory')
    parser.add_argument('--dataset', type=str, required=True, help='json dataset file')
    parser.add_argument('--result_path', type=str, required=True, help='json dataset file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for inference')
    parser.add_argument("--temp_dir", type=str, default="tmp")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--skip_len", type=int, default=1000)
    
    args = parser.parse_args()
    
    infer(args)