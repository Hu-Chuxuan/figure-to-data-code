import cv2
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64

MIN_LINE = 50
MIN_GAP = 5
MIN_TICK_LEN = 5
MAX_TICK_LEN = 50
MAX_LINE_WIDTH = 10
MAX_ALLOWED_GAP = 5

def encode_image(image):
    # use base64 to encode the cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class Axis:
    def __init__(self, line_r, line_c_lo, line_c_hi, direction, color):
        self.direction = direction
        self.line_r_lo = line_r
        self.line_r_hi = line_r+1
        self.line_c_lo = line_c_lo
        self.line_c_hi = line_c_hi
        self.color = color
        self.ticks = None
        self.labels = None
    
    def merge(self, axis):
        self.line_r_lo = min(self.line_r_lo, axis.line_r_lo)
        self.line_r_hi = max(self.line_r_hi, axis.line_r_hi)
        self.line_c_lo = min(self.line_c_lo, axis.line_c_lo)
        self.line_c_hi = max(self.line_c_hi, axis.line_c_hi)

class Subplot:
    def __init__(self, x, y):
        self.x_axis = x
        self.y_axis = y

class SubplotConstructor:
    def __init__(self, api_key=None, organization=None):
        self.bg_color = (255, 255, 255)
        self.client = OpenAI(api_key=api_key, organization=organization)
    
    def filter_axes(self, image, potential_axes):
        axes = []
        for axis in potential_axes:
            if axis.line_r_hi - axis.line_r_lo > MAX_LINE_WIDTH:
                continue
            sting_pos = {}
            for i in range(axis.line_c_lo, axis.line_c_hi):
                for j in range(axis.line_r_hi, image.shape[0]):
                    if np.array_equal(image[j][i], self.bg_color) or \
                       (j - axis.line_r_hi+1 > MIN_TICK_LEN and not np.array_equal(image[j][i], axis.color)) or \
                       not np.array_equal(image[j][i], axis.color):
                        if axis.line_r_lo == 902 and 775 <= i <= 779:
                            print(i, j-axis.line_r_hi+1)
                        if MIN_TICK_LEN < j - axis.line_r_hi + 1 < MAX_TICK_LEN:
                            sting_pos[i] = 1
                        break
                for j in range(axis.line_r_lo-1, 0, -1):
                    if np.array_equal(image[j][i], self.bg_color) or (axis.line_r_lo - j - 1 > MIN_TICK_LEN and not np.array_equal(image[j][i], axis.color)) or not np.array_equal(image[j][i], axis.color):
                        if MIN_TICK_LEN < axis.line_r_lo - j - 1 < MAX_TICK_LEN:
                            sting_pos[i] = 1
                        break
            if axis.line_r_lo == 902:
                print(sting_pos.keys())
            if len(sting_pos) == 1:
                continue
            grouped_stings = []
            for sting in sorted(sting_pos.keys()):
                if len(grouped_stings) == 0 or sting - grouped_stings[-1][-1] > MIN_GAP:
                    grouped_stings.append([sting])
                else:
                    grouped_stings[-1].append(sting)
            ticks = []
            for group in grouped_stings:
                if group[-1] - group[0] < MIN_GAP:
                    ticks.append((group[0] + group[-1]) / 2)

            if axis.line_r_lo == 902:
                print(grouped_stings)
                print(ticks)

            if len(ticks) > 0:
                axis.ticks = ticks
                axes.append(axis)
        return axes

    def merge_straight_lines(self, image, potential_axes):
        # Can this line be merged with an existing axis
        merged_lines = []
        for i in range(len(potential_axes)):
            merged = False
            for axis in merged_lines:
                if not np.array_equal(axis.color, potential_axes[i].color):
                    continue
                to_merge = False
                if axis.line_c_lo <= potential_axes[i].line_c_hi and potential_axes[i].line_c_lo <= axis.line_c_hi:
                    # If the two lines overlap at the column direction
                    to_merge = True
                    for k in range(max(axis.line_c_lo, potential_axes[i].line_c_lo), min(axis.line_c_hi, potential_axes[i].line_c_hi)):
                        allowed_gap = MAX_ALLOWED_GAP
                        for l in range(min(axis.line_r_hi, potential_axes[i].line_r_hi), max(axis.line_r_lo, potential_axes[i].line_r_lo)):
                            if not np.array_equal(image[l][k], axis.color):
                                if allowed_gap == 0:
                                    to_merge = False
                                    break
                                else:
                                    allowed_gap -= 1
                            else:
                                allowed_gap = MAX_ALLOWED_GAP
                        if not to_merge:
                            break
                elif axis.line_r_lo <= potential_axes[i].line_r_hi and potential_axes[i].line_r_lo <= axis.line_r_hi:
                    # If the two lines overlap at the row direction
                    # They are definitely not overlapped at the column direction
                    to_merge = True
                    for k in range(max(axis.line_r_lo, potential_axes[i].line_r_lo), min(axis.line_r_hi, potential_axes[i].line_r_hi)):
                        allowed_gap = MAX_ALLOWED_GAP
                        for l in range(min(axis.line_c_hi, potential_axes[i].line_c_hi), max(axis.line_c_lo, potential_axes[i].line_c_lo)):
                            if not np.array_equal(image[k][l], axis.color):
                                if allowed_gap == 0:
                                    to_merge = False
                                    break
                                else:
                                    allowed_gap -= 1
                            else:
                                allowed_gap = MAX_ALLOWED_GAP
                if to_merge:
                    axis.merge(potential_axes[i])
                    merged = True
                    break
            if not merged:
                merged_lines.append(potential_axes[i])

        return merged_lines

    def estimate_ticks(self, image, direction):
        if direction == "y":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # identify straight lines
        potential_axes = []
        for i in tqdm(range(image.shape[0])):
            cur_color = image[i][0]
            cur_line_lo = 0
            j = 1
            while j <= image.shape[1]:
                if j == image.shape[1] or not np.array_equal(image[i][j], cur_color):
                    if not np.array_equal(cur_color, self.bg_color) and j - cur_line_lo > MIN_LINE and i > 0:
                        # Different to majority of at least one of the two sides to be an edge
                        upper_diff = 0
                        lower_diff = 0
                        for k in range(cur_line_lo, j):
                            if i > 0 and not np.array_equal(image[i-1][k], cur_color):
                                upper_diff += 1
                            if i < image.shape[0] and not np.array_equal(image[i+1][k], cur_color):
                                lower_diff += 1
                        if upper_diff / (j - cur_line_lo) > 0.75 or lower_diff / (j - cur_line_lo) > 0.75:
                            potential_axes.append(Axis(i, cur_line_lo, j, direction, cur_color))
                    if j < image.shape[1]:
                        cur_color = image[i][j]
                    cur_line_lo = j
                j += 1
        potential_axes = self.merge_straight_lines(image, potential_axes)
        
        axes = self.filter_axes(image, potential_axes)
        if direction == "y":
            for axis in axes:
                axis.line_c_lo, axis.line_c_hi = image.shape[1]-axis.line_c_hi, image.shape[1]-axis.line_c_lo
                for i in range(len(axis.ticks)):
                    axis.ticks[i] = image.shape[1] - axis.ticks[i]
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return axes
    
    def draw_axes(self, image, axes):
        for i, axis in enumerate(axes):
            text = axis.direction+"["+str(i)+"]"
            height, width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            if axis.direction == "x":
                cv2.line(image, (axis.line_c_lo, axis.line_r_lo), (axis.line_c_hi, axis.line_r_lo), (0, 0, 255), 2)
                cv2.putText(image, axis.direction+"["+str(i)+"]", (axis.line_c_hi+height, axis.line_r_lo+width), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.line(image, (axis.line_r_lo, axis.line_c_lo), (axis.line_r_lo, axis.line_c_hi), (255, 0, 0), 2)
                cv2.putText(image, axis.direction+"["+str(i)+"]", (axis.line_r_lo+height, axis.line_c_lo+width), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return image

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
