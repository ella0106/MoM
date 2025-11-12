from utils.utils import *
from dataset import CustomDataset, DataCollatorForCustomDataset
from builder import get_model
from utils.constants import IGNORE_INDEX
from transformers import Trainer, TrainingArguments

def train(args):
    rank0_print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    tokenizer, model, max_length = get_model(args)
    model.train()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.config.use_reentrant = False  
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    rank0_print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    rank0_print("Model max length:", max_length)
    
    train_dataset = CustomDataset(
        video_dir=args.video_dir,
        txt=args.train_ann,
        temp_dir=args.temp_dir,
        tokenizer=tokenizer,
        max_len=max_length,
        conv_template="qwen_1_5"
    )
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = DataCollatorForCustomDataset(pad_token_id=pad_id, ignore_index=IGNORE_INDEX)
     
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed_config,
        fp16=True,
        save_total_limit=2,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        data_collator=collator, 
        )
    
    if args.resume_checkpoint is not None:
        rank0_print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    else:
        trainer.train()
    
    trainer.save_model(args.output_dir)    
    
        
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument('--model_type', type=str, required=True, help='pretrained model type')
    p.add_argument('--tokenizer_path', type=str, required=True, help='tokenizer path')
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--train_ann", type=str, required=True)
    p.add_argument("--temp_dir", type=str, default="tmp")
    p.add_argument("--output_dir", type=str, default="./outputs_vqa")
    p.add_argument("--resume_checkpoint", type=str, default=None)

    p.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100, help="저장 간격(step 단위)")
    p.add_argument("--run_name", type=str, default="MoM-training")
    
    p.add_argument("--local_rank", type=int, default=-1, help="deepspeed에서 자동으로 할당")
    
    args = p.parse_args()
    
    train(args)