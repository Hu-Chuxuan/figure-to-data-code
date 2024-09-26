import cv2
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64

def encode_image(image):
    # use base64 to encode the cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class Plot:
    def __init__(self, api_key=None, organization=None):
        self.client = OpenAI(api_key=api_key, organization=organization)

    def construct_subplot(self, image):
        x = self.estimate_ticks(image, "x")
        y = self.estimate_ticks(image, "y")
        image = self.draw_axes(image, x)
        image = self.draw_axes(image, y)
        cv2.imwrite("Tests/Actual/axis_cluster_sharped.png", image)
        prompt = '''
Now, you are given a picture of chart with potential axes detected. Your task is to construct the subplots based on the detected axes. 

In the first step, you need to identify the number of subplots in the chart. Then, for each subplot, you need to identify its x-axis and y-axis. Your output should a several lines of Python code that can be executed to generate the subplots. You are provided with the function Subplot(x, y) to construct a subplot with one x-axis x and one y-axis y. 

For example, you have an image with two subplots, while the labeled axes are x-axis x[0] and x[1], and y-axis y[0] and y[1], y[2], y[3]. If the first plot are constructed with x[0], y[1], while the second one are contructed with x[1], y[0], you should output:
```python
subplots.append(Subplot[x[0], y[1]))
subplots.append(Subplot[x[1], y[0]))
```

Note that not all labeled axes are actually used in the subplots. You need to figure out which axes are used in which subplot and which are not. You should only give one code block in the output correctly construct the subplots you have identified. 
'''
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
            model="gpt-4o",
            messages=msg,
            temperature=0.7,
            max_tokens=4096,
            top_p=1
        )
        print(response.choices[0].message.content)
        print("# of input tokens: ", response.usage.prompt_tokens)
        print("# of output tokens: ", response.usage.completion_tokens)
        
        subplots = []
        start = response.choices[0].message.content.find("```python\n")
        ends = response.choices[0].message.content.find("```", start+1)
        if response.choices[0].message.content.find("```python\n", ends+1) != -1:
            print("Multiple code blocks detected")
        else:
            exec(response.choices[0].message.content[start+9:ends])
            return subplots
