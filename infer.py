from utils.utils import *
from dataset import CustomDataset, DataCollatorForCustomDataset
from builder import load_pretrained_model
from utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from utils.conversation import conv_templates
from transformers import GenerationConfig

def to_device(batch, device):
    """
    배치를 주어진 device로 안전하게 이동.
    input_ids, labels, attention_mask, images만 사용.
    (motion_feats, residual_feats는 현재 모델 시그니처상 전달하지 않음)
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            # 멀티모달 텐서는 AMP dtype으로
            if k in ("images", "motion_feats", "residual_feats"):
                out[k] = v.to(device=device, dtype=amp_dtype, non_blocking=True)
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 경우
    # cudnn 관련 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 환경 변수 (일부 라이브러리에서 사용)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed fixed to {seed}]")

# 예시
set_seed(2025)

def main(args):
    model_name = args.model_path
    model_type = "llava_qwen_mom"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_name, None, model_type, torch_dtype="float16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()

    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"[NaN Fix] {name} contained NaN → replaced with 0")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
    
    
    dataset = CustomDataset(
        video_dir=args.video_dir,
        txt=args.dataset,
        tokenizer=tokenizer,
        max_len=max_length,
        conv_template="qwen_1_5"
    )
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = DataCollatorForCustomDataset(pad_token_id=pad_id, ignore_index=IGNORE_INDEX)
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # DS에서 실제 마이크로 배치는 config로도 관리됨
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator
    )
    
    gen_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        temperature=1,
    )
    
    results = {}
    for i, batch in enumerate(tqdm(loader)):
        batch = to_device(batch, model.device)
        if batch["input_ids"].dim() == 3 and batch["input_ids"].size(0) == 1:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
                batch["labels"] = batch["labels"].squeeze(0)
                batch["attention_mask"] = batch["attention_mask"].squeeze(0)
        
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
        if i % 10 == 0:
            save_file([results], args.result_path)
    
    save_file([results], args.result_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', required=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='pretrained model path')
    parser.add_argument('--video-dir', type=str, required=True, help='video directory')
    parser.add_argument('--dataset', type=str, required=True, help='json dataset file')
    parser.add_argument('--result-path', type=str, required=True, help='json dataset file')

    args = parser.parse_args()
    
    main(args)