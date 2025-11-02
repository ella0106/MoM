from utils.utils import *
from utils.conversation import conv_templates
from processor import MotionVectorExtractor, MotionFeatureExtractor
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, video_dir, txt, temp_dir, tokenizer=None, max_len=None, conv_template=None, train=True):
        self.video_dir = video_dir
        self.data = load_file(txt)
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.conv_template = conv_template
        self.mve = MotionVectorExtractor(temp_dir=temp_dir)
        self.mfe = MotionFeatureExtractor()
        sefl.train = train

    def __len__(self):
        return len(self.data)
    
    def _build_prompt(self, question):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        question = DEFAULT_IMAGE_TOKEN + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __getitem__(self, idx):
        cur_sample = self.data[idx]
                     
        video_path = os.path.join(self.video_dir, cur_sample["video"])
        try:
            frames, motions_norm, motion_indices, motions_raw = self.mve(video_path)   # 프레임, 모션, 인덱스 추출
            if frames is None:
                raise ValueError("Invalid Video")
            images, motion_feats, residual_feats = self.mfe(frames, motions_norm, motion_indices, motions_raw)
            
            question = cur_sample["question"]
            candidates = cur_sample["candidates"]
            answer = cur_sample["answer"]
            candidates_str = " ".join(candidates)
            question = ("The video frames are extracted at 2 fps, followed by auxiliary motion tokens that can be used as additional cues. Please answer the following questions related to this video.\n"
                        f"Question: {question} Options: {candidates_str}\nSelect the best option to answer the question."
            )
            prompt = self._build_prompt(question)
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            prompt_len = len(input_ids)
            
            if self.train:
                answer_ids = self.tokenizer(
                    answer + self.tokenizer.eos_token,
                    return_tensors="pt",
                    add_special_tokens=False
                )["input_ids"][0]
                input_ids = torch.cat([input_ids, answer_ids], dim=0)
                input_ids = input_ids[: self.max_length]
            
            labels = input_ids.clone()
            # input_ids[prompt_len:] = IGNORE_INDEX
            labels[:prompt_len] = IGNORE_INDEX
            
            return {
                "input_ids": input_ids,
                "labels": labels,
                "images" : images,
                "motion_feats" : motion_feats,
                "residual_feats" : residual_feats,
                "modalities" : "video"
            }
    
        except Exception as e:
            print(f"[⚠️ Warning] Index {idx} Video : {video_path} 처리 중 오류 발생 → 건너뜀 ({e})")
            next_idx = (idx + 1) % len(self.data)
            return self.__getitem__(next_idx)
    
class DataCollatorForCustomDataset:
    def __init__(self, pad_token_id, ignore_index=-100, train=True):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.train = train

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        images = [b["images"] for b in batch]
        motion_feats = [b["motion_feats"] for b in batch]
        residual_feats = [b["residual_feats"] for b in batch]
        modalities = [b["modalities"] for b in batch]
        
        if self.train:
            labels = torch.full((len(batch), max_len), self.ignore_index, dtype=torch.long)

            for i, b in enumerate(batch):
                l = len(b["input_ids"])
                input_ids[i, :l] = b["input_ids"]
                labels[i, :l] = b["labels"]
                attn[i, :l] = 1
        
        else:
            labels = None
            
            for i, b in enumerate(batch):
                l = len(b["input_ids"])
                input_ids[i, :l] = b["input_ids"]
                attn[i, :l] = 1
                

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "images" :  images,
            "motion_feats" : motion_feats,
            "residual_feats" : residual_feats,
            "modalities" : modalities,
        }
        
if __name__ == "__main__":
    dataset = CustomDataset(
        video_dir='dataset/Anet/',
        temp_dir='tmp/',
        txt='dataset/train.json',
        tokenizer=AutoTokenizer.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2"),
        max_len=10024,
        conv_template="qwen_1_5"
    )
    
    for idx in tqdm(range(len(dataset))):
        if idx < 32100:
            continue
        sample = dataset[idx]