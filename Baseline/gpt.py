import cv2
from openai import OpenAI
import anthropic
from io import BytesIO
from PIL import Image
import base64
import argparse
import pandas as pd
from io import StringIO

def encode_image(image):
    # use base64 to encode the cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def gpt(prompt, image, api_key=None, organization=None, model="gpt-4o"):
    client = OpenAI(api_key=api_key, organization=organization)
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

    response = client.chat.completions.create(
        model=model,
        messages=msg,
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
    )
    print(response.choices[0].message.content)
    print("# of input tokens: ", response.usage.prompt_tokens)
    print("# of output tokens: ", response.usage.completion_tokens)

    response = response.choices[0].message.content
    return response

def claude(prompt, image, api_key=None, organization=None, model="claude-3-5-sonnet-20241022"):
    client = anthropic.Client(api_key=api_key)
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
    return response.content[0].text

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

def digitize(prompt, image, api_key=None, org=None, model="gpt-4o"):
    if "claude" in model:
        response = claude(prompt, image, api_key, org)
    else:
        response = gpt(prompt, image, api_key, org, model)
    res = parse_response(response)
    return res, response

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

