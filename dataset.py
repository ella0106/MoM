from utils.utils import *
from utils.conversation import conv_templates
from processor import MotionVectorExtractor, MotionFeatureExtractor
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, video_dir, txt, temp_dir, tokenizer=None, conv_template=None, train=True, gop_num=8, fps=4, image_processor=None, config=None):
        self.video_dir = video_dir
        self.data_name = txt
        self.data = load_file(txt)
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.train = train
        self.gop_num = gop_num
        self.mve = MotionVectorExtractor(temp_dir=temp_dir, fps=fps)
        self.mfe = MotionFeatureExtractor(frame_num=fps*2, config=config)
        self.image_processor = image_processor
        self.config = config

    def __len__(self):
        return len(self.data)
    
    def _build_prompt(self, question):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        question = DEFAULT_IMAGE_TOKEN + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __getitem__(self, idx):
        if hasattr(self.data, "iloc"):  # pandas DataFrame (parquet)
            cur_sample = self.data.iloc[idx]
        else:
            cur_sample = self.data[idx]
                     
        video_path = os.path.join(self.video_dir, cur_sample["video"])
        try:
            frames, motions_norm, motion_indices, motions_raw = self.mve(video_path)   # 프레임, 모션, 인덱스 추출
            if frames is None:
                raise ValueError("Invalid Video")
            pixels, motion_feats, residual_feats = self.mfe(frames, motions_norm, motion_indices, motions_raw)
            if pixels.shape[0] > self.gop_num:
                T = pixels.shape[0]
                selected_indices = torch.linspace(0, T-1, steps=self.gop_num).long()
                pixels = pixels[selected_indices]
                
                if motion_feats is not None:
                    motion_feats = motion_feats[selected_indices]
                if residual_feats is not None:
                    residual_feats = residual_feats[selected_indices]
            images = self.image_processor.preprocess(pixels, return_tensors="pt")["pixel_values"]

            if "next" in self.video_dir.lower():
                prompt, label = self.process_nextqa(cur_sample)
            if "anet" in self.video_dir.lower() or "activitynet" in self.video_dir.lower():
                if "caption" in self.data_name.lower():
                    prompt, label, timestamp = self.process_anet_caption(cur_sample)
                    start, end = timestamp
                    images, motion_feats, residual_feats = images[start:end+1], motion_feats[start:end+1], residual_feats[start:end+1]
                if "qa" in self.data_name.lower():
                    prompt, label = self.process_anet_qa(cur_sample)
            if "msrvtt" in self.video_dir.lower() or "msvd" in self.video_dir.lower():
                prompt, label = self.process_msrvtt(cur_sample)
            if "videomme" in self.video_dir.lower():
                prompt, label = self.process_videomme(cur_sample)
            else:
                prompt = cur_sample['question']
                label = cur_sample['answer']

            prompt = self._build_prompt(prompt)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            prompt_len = len(input_ids)
            
            if self.train:
                label_ids = self.tokenizer(
                    label + self.tokenizer.eos_token,
                    return_tensors="pt",
                    add_special_tokens=False
                )["input_ids"][0]
                input_ids = torch.cat([input_ids, label_ids], dim=0)
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
            # next_idx = (idx + 1) % len(self.data)
            # return self.__getitem__(next_idx)
            return None
  
    def process_nextqa(self, cur_sample):
        question = cur_sample["question"]
        candidates = cur_sample["candidates"]
        answer = cur_sample["answer"]
        candidates_str = " ".join(candidates)
        question = ("The video frames are extracted at 2 fps, followed by auxiliary motion tokens that can be used as additional cues. Please answer the following questions related to this video.\n"
                    f"Question: {question} Options: {candidates_str}\nSelect the best option to answer the question."
        )
        return question, answer
    
    def process_anet_caption(self, cur_sample):
        question = "The video frames are extracted at 2 fps, followed by auxiliary motion tokens that can be used as additional cues. \
                    Describe the content of the video."
        answer = cur_sample["sentence"]
        
        def floor_even(t):
            t_floor = math.floor(t)
            return t_floor if t_floor % 2 == 0 else t_floor - 1
        
        timestamp = [floor_even(x)//2 for x in cur_sample['timestamp']]

        return question, answer, timestamp
    
    def process_anet_qa(self, cur_sample):
        question = cur_sample['question'].capitalize() + "?"
        question = (f"{question} Answer the question using a single word or phrase.")
        answer = " "
        return question, answer
    
    def process_msrvtt(self, cur_sample):
        question = cur_sample['question'].capitalize()
        question = (f"{question} Answer the question using a single word or phrase.")
        answer = cur_sample['answer']
        return question, answer
    
    def process_videomme(self, cur_sample):
        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
        question = cur_sample["question"]
        option = "\n".join([f"{opt}" for i, opt in enumerate(cur_sample["options"])])
        question = question + "\n" + option
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + post_prompt

        answer = cur_sample['answer']
        return full_prompt, answer
        
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
        
class BaseDataset(Dataset):
    def __init__(self, video_dir, txt, temp_dir, tokenizer=None, max_len=None, conv_template=None, image_processor=None):
        self.video_dir = video_dir
        self.data_name = txt
        self.data = load_file(txt)
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.conv_template = conv_template
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.data)
    
    def _build_prompt(self, question, video, frame_time, video_time):
        # time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"{question}"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def __getitem__(self, idx):
        cur_sample = self.data[idx]
                     
        video_path = os.path.join(self.video_dir, cur_sample["video"])
        video,frame_time,video_time = load_video(video_path)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()

        if "next" in self.video_dir.lower():
                prompt, label = self.process_nextqa(cur_sample)
        if "anet" in self.video_dir.lower():
            if "caption" in self.data_name.lower():
                prompt, label, timestamp = self.process_anet_caption(cur_sample)
                start, end = timestamp
            if "qa" in self.data_name.lower():
                prompt = cur_sample['question'].capitalize() + "? Answer the question using a single word or phrase."
        
        prompt = self._build_prompt(prompt, video, frame_time, video_time)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prompt_len = len(input_ids)
        
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "images" : video,
            "modalities" : "video"
        }
    
class DataCollatorForBaseDataset:
    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        images = [b["images"] for b in batch]
        modalities = [b["modalities"] for b in batch]
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
            "modalities" : modalities,
        }  

class EMADataset(Dataset):
    def __init__(self, video_dir, txt, temp_dir, tokenizer=None, conv_template=None, train=True, gop_num=8, fps=4, image_processor=None, config=None):
        self.video_dir = video_dir
        self.data_name = txt
        self.data = load_file(txt)
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.train = train
        self.gop_num = gop_num
        self.mve = MotionVectorExtractor(temp_dir=temp_dir, fps=fps)
        self.mfe = MotionFeatureExtractor(frame_num=fps*2, config=config)
        self.image_processor = image_processor
        self.config = config
        
    def __len__(self):
        return len(self.data)
    
    def _build_prompt(self, question):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def __getitem__(self, idx):
        cur_sample = (
            self.data.iloc[idx]
            if hasattr(self.data, "iloc")
            else self.data[idx]
        )
        
        try:
                        
            if "video" in cur_sample and cur_sample["video"] is not None:
                modality = "video"
                video_path = os.path.join(self.video_dir, "videos", cur_sample["video"])
                frames, motions_norm, motion_indices, motions_raw = self.mve(video_path)   # 프레임, 모션, 인덱스 추출
                if frames is None:
                    raise ValueError("Invalid Video")
                pixels, motion_feats, residual_feats = self.mfe(frames, motions_norm, motion_indices, motions_raw)
                if pixels.shape[0] > self.gop_num:
                    T = pixels.shape[0]
                    selected_indices = torch.linspace(0, T-1, steps=self.gop_num).long()
                    pixels = pixels[selected_indices]
                    
                    if motion_feats is not None:
                        motion_feats = motion_feats[selected_indices]
                    if residual_feats is not None:
                        residual_feats = residual_feats[selected_indices]
                pixels = self.image_processor.preprocess(pixels, return_tensors="pt")["pixel_values"]

            elif "image" in cur_sample and cur_sample["image"] is not None:
                modality = "image"
                image_path = os.path.join(self.video_dir, "images", cur_sample["image"])
                pixels = load_image(image_path)
                pixels = self.image_processor.preprocess(pixels, return_tensors="pt")["pixel_values"]
                motion_feats = torch.zeros(1, self.gop_num, 2, 96, 96).bfloat16()
                residual_feats = None

            else:
                return None
                    
            convs = cur_sample["conversations"]
            question = convs[0]['value']
            label = convs[1]['value']

            prompt = self._build_prompt(question)
            input_ids = fast_tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, modality)
            prompt_len = len(input_ids)
            
            if self.train:
                label_ids = self.tokenizer(label + self.tokenizer.eos_token, return_tensors="pt", add_special_tokens=False )["input_ids"][0] 
                input_ids = torch.cat([input_ids, label_ids], dim=0) 
                
            labels = input_ids.clone()
            labels[:prompt_len] = IGNORE_INDEX
        
            return {
                "input_ids": input_ids,
                "labels": labels,
                "images" : pixels,
                "motion_feats" : motion_feats,
                "residual_feats" : residual_feats,
                "modalities" : modality
            }
        
        except Exception as e:
            print(f"[⚠️ Warning] Index {idx} Video : {video_path} 처리 중 오류 발생 → 건너뜀 ({e})")
            return None
        
class PackedEMADataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        max_length=4096,
        packed_indices=None,
    ):
        self.raw_dataset = raw_dataset
        self.max_length = max_length
        self.packed_indices = packed_indices

    def __len__(self):
        return len(self.packed_indices)
    
    def __getitem__(self, idx):
        indices = self.packed_indices[idx]
        
        input_ids_list = []
        labels_list = []
        images_list = []
        motion_feats_list = []
        residual_feats_list = []
        modalities_list = []
        total_length = 0

        for i in indices:
            sample = self.raw_dataset[i]
            if sample is None:
                continue
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            length = len(input_ids)

            if total_length + length <= self.max_length:
                input_ids_list.append(input_ids)
                labels_list.append(labels)
                images_list.append(sample["images"])
                motion_feats_list.append(sample["motion_feats"])
                residual_feats_list.append(sample["residual_feats"])
                modalities_list.append(sample["modalities"])
                total_length += length
            else:
                break

        if total_length == 0:
            return None

        packed_input_ids = torch.cat(input_ids_list, dim=0)
        packed_labels = torch.cat(labels_list, dim=0)

        return {
            "input_ids": packed_input_ids,
            "labels": packed_labels,
            "images" : images_list,
            "motion_feats" : motion_feats_list,
            "residual_feats" : residual_feats_list,
            "modalities" : modalities_list,
        }

class EMACollator:
    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        B = len(batch)
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids = torch.full(
            (B, max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full(
            (B, max_len), self.ignore_index, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (B, max_len), dtype=torch.bool
        )

        batch_images = []
        batch_motion = []
        batch_residual = []
        batch_modalities = []

        for i, b in enumerate(batch):
            l = len(b["input_ids"])
            input_ids[i, :l] = b["input_ids"]
            labels[i, :l] = b["labels"]
            attention_mask[i, :l] = True

            batch_images.append(b["images"])
            batch_motion.append(b["motion_feats"])
            batch_residual.append(b["residual_feats"])
            batch_modalities.append(b["modalities"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": batch_images,
            "motion_feats": batch_motion,
            "residual_feats": batch_residual,
            "modalities": batch_modalities,
        }


      
if __name__ == "__main__":
    from transformers import SiglipImageProcessor

    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        
    dataset = EMADataset(
        video_dir='dataset/pretrain/',
        temp_dir='/mnt/aix21002/MoM/tmp/',
        txt='dataset/pretrain/pretrain.parquet',
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2-7B"),
        conv_template="qwen_1_5",
        image_processor=processor,
    )
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]