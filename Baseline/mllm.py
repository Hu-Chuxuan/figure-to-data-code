import cv2
from openai import OpenAI
import anthropic
from io import BytesIO
from PIL import Image
import base64
import argparse
import pandas as pd
import torch
from io import StringIO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle

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

        response = client.messages.create(
            model=model,
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
        )

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
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
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
        self.model.eval()

    def query(self, prompt, image_path):
        image = Image.open(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "qwen_2"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs)

        return text_outputs, parse_response(text_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Digitize a plot or a table from an image')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument("--output", type=str, help="Path to the output CSV file")
    parser.add_argument('--api', type=str, help='OpenAI API key')
    parser.add_argument('--org', type=str, help='OpenAI organization')

    args = parser.parse_args()

    image = cv2.imread(args.image)
    res, response = digitize(image, args.api, args.org)
    for i, df in enumerate(res):
        if len(res) == 1:
            df.to_csv(args.output, index=False)
        else:
            df.to_csv(f"{args.output[:-4]}-{i}.csv", index=False)
    with open(f"{args.output[:-4]}.txt", "w") as f:
        f.write(response)

