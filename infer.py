from llava.mom.utils.utils import *
from llava.mom.dataset import *
from llava.mom.builder import load_pretrained_model
from llava.mom.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.mom.utils.conversation import conv_templates
from transformers import GenerationConfig

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
    )
    
    gen_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        temperature=1,
    )
    
    results = {}
    for i, (video_path, question, _) in enumerate(tqdm(dataset)):
        # print(video_path, question)
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + question
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        cont = model.generate(
            input_ids,
            images=video_path,
            modalities= ["video"],
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