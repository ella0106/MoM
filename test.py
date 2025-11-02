from utils.utils import *
from dataset import CustomDataset, DataCollatorForCustomDataset 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")
max_length = 2048
 
train_dataset = CustomDataset(
        video_dir='dataset/NExTVideo/',
        txt='dataset/train.json',
        temp_dir='tmp/',
        tokenizer=tokenizer,
        max_len=max_length,
        conv_template="qwen_1_5"
    ) 

for i in range(1):
    print(type(train_dataset[i]["input_ids"]), train_dataset[i]["input_ids"])
    inputs = train_dataset[i]["input_ids"].tolist()[15:]
    print(inputs)
    print("Label:", train_dataset[i]["labels"])
