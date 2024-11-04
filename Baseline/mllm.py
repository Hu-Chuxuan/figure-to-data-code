import cv2
from openai import OpenAI
import anthropic
import copy, math
from io import BytesIO
from PIL import Image
import base64
import argparse
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from io import StringIO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModel
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

def encode_image(image):
    # use base64 to encode the cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def parse_response(response):
    res = []
    pos = 0
    while pos < len(response):
        start = response.find("```csv", pos)
        if start == -1:
            break
        end = response.find("```", start+7)
        df = pd.read_csv(StringIO(response[start+7:end]))
        res.append(df)
        pos = end + 3

    return res

class GPT:
    def __init__(self, api, org, model):
        self.client = OpenAI(api_key=api, organization=org)
        self.model = model
    
    def query(self, prompt, image_path):
        image = cv2.imread(image_path)
        encoded_img = encode_image(image)
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_img}",
                        },
                    },
                ]
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msg,
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
        )
        print(response.choices[0].message.content)
        print("# of input tokens: ", response.usage.prompt_tokens)
        print("# of output tokens: ", response.usage.completion_tokens)

        response = response.choices[0].message.content
        return response, parse_response(response)

class Claude:
    def __init__(self, api, org, model):
        self.client = anthropic.Client(api_key=api, organization=org)
        self.model = model

    def query(self, prompt, image_path):
        image = cv2.imread(image_path)
        encoded_img = encode_image(image)
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_img,
                        },
                    },
                ]
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=msg,
        )
        print(response.content[0].text)
        print("# of input tokens: ", response.usage.input_tokens)
        print("# of output tokens: ", response.usage.output_tokens)
        
        return response.content[0].text, parse_response(response.content[0].text)

class Qwen:
    def __init__(self, model="Qwen/Qwen2-VL-72B-Instruct"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto",
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(model)

    def query(self, prompt, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image": image_path,
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(output_text)

        return output_text, parse_response(output_text)

class Molmo:
    def __init__(self, model='allenai/Molmo-72B-0924'):
        self.processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
    
    def query(self, prompt, image_path):
        inputs = self.processor.process(
            images=[Image.open(image_path)],
            text=prompt,
        )
        inputs["images"] = inputs["images"].to(torch.bfloat16)

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(generated_text)

        return generated_text, parse_response(generated_text)

class LLAVA:
    def __init__(self, model):
        model_name = "llava_qwen"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                model, None, model_name, device_map="auto")
        # self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
        #         model, None, model_name, device_map="auto")
        self.model.eval()

    def query(self, prompt, image_path):
        image = Image.open(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]

        conv_template = "qwen_2"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        # input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        print(text_outputs)

        return text_outputs, parse_response(text_outputs)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVL:
    def __init__(self, model='InternVL2-Llama3-76B'):
        device_map = split_model(model)
        self.model = AutoModel.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)

    def query(self, prompt, image_path):
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        response = self.model.chat(self.tokenizer, pixel_values, '<image>\n'+prompt, generation_config)
        print(response)
        return response, parse_response(response)
