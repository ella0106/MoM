from llava.mom.utils.utils import *
from llava.mom.dataset import CustomDataset, DataCollatorForCustomDataset
from llava.mom.builder import get_model
from llava.mom.utils.constants import IGNORE_INDEX
from transformers import get_linear_schedule_with_warmup
amp_dtype = torch.float16

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
    
def build_dataloader(args, tokenizer, max_length):
    
    dataset = CustomDataset(
        video_dir=args.video_dir,
        txt=args.train_ann,
        tokenizer=tokenizer,
        max_len=max_length,
        conv_template="qwen_1_5"
    )
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = DataCollatorForCustomDataset(pad_token_id=pad_id, ignore_index=IGNORE_INDEX)
    
    loader = DataLoader(
        dataset,
        batch_size=args.train_micro_batch_size,  # DS에서 실제 마이크로 배치는 config로도 관리됨
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collator
    )
    
    return loader

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

def save_checkpoint_step(model_engine, tokenizer, model, save_dir, global_step):
    """
    step 기준으로 저장 (rank0만)
    """
    if model_engine.global_rank == 0:
        step_dir = os.path.join(save_dir, f"step-{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        model_engine.save_checkpoint(step_dir)
        try:
            model.save_pretrained(step_dir)
            tokenizer.save_pretrained(step_dir)
        except Exception:
            pass
        print(f"💾 Checkpoint saved at step {global_step} -> {step_dir}")

def train(args):
    set_seed(2025)
    tokenizer, model, image_processor, max_length = get_model(args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # print("--- Trainable Parameters ---")
    # total_trainable_params = 0
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer: {name} | Size: {param.size()} | Num_elems: {param.numel()}")
    #         total_trainable_params += param.numel()

    # print(f"\nTotal trainable parameters: {total_trainable_params}")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    
    train_loader = build_dataloader(args, tokenizer, max_length)

    total_steps = math.ceil(len(train_loader) * args.epochs / max(1, args.gradient_accumulation_steps))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    model_engine, optimizer_engine, _, scheduler_engine = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )

    model_engine.train()
    global_step = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            # ========= 배치 전처리 =========
            # CustomDataset이 반환하는 키: input_ids, labels, images, (motion_feats, residual_feats)
            # collator가 attention_mask를 만들어주므로 그것도 같이 사용
            # 일부 구현에서 input_ids shape가 (1, L) 문제가 있으면 squeeze 처리
            if batch["input_ids"].dim() == 3 and batch["input_ids"].size(0) == 1:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
                batch["labels"] = batch["labels"].squeeze(0)
                batch["attention_mask"] = batch["attention_mask"].squeeze(0)

            batch = to_device(batch, model_engine.device)

            # 현재 모델 시그니처는 images만 받음. motion/residual은 사용 X
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["images"],
                labels=batch["labels"],
                motion_feats=batch["motion_feats"],
                residual_feats=batch["residual_feats"],
                modalities=batch["modalities"],
            )
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
            scheduler_engine.step()

            if (global_step % args.log_steps == 0) and (model_engine.global_rank == 0) and (global_step > 0):
                lr = scheduler_engine.get_lr()[0]
                print(f"[epoch {epoch+1}] step {global_step} | loss {loss.item():.4f} | lr {lr:.6e}")

            global_step += 1

        if global_step % args.save_steps == 0:
            save_checkpoint_step(model_engine, tokenizer, model, args.output_dir, global_step)

    if model_engine.global_rank == 0:
        print("✅ Training completed.")    
    
    
        
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--train_ann", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs_vqa")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--train_micro_batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100, help="저장 간격(step 단위)")
    p.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1, help="(automatically passed by deepspeed/torchrun)")
    p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--master_addr", type=str)
    
    args = p.parse_args()
    
    train(args)