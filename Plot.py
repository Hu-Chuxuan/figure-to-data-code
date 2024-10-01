import cv2
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
from Subplot import DotPlot

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
        self.subplots = []
    
    def appendDotPlot(self, x, y, subplot_value, has_error_bars, value_direction):
        self.subplots.append(DotPlot(x, y, subplot_value, has_error_bars, value_direction))

    def construct_subplot(self, image):
        x = self.estimate_ticks(image, "x")
        y = self.estimate_ticks(image, "y")
        image = self.draw_axes(image, x)
        image = self.draw_axes(image, y)
        cv2.imwrite("Tests/Actual/axis_cluster_sharped.png", image)
    
    def gpt(self, image):
        prompt = '''
### Task: Construct Subplots from Detected Axes in a Chart

You are given a chart image where potential axes have been detected, and lines in the image are labeled with indices. Your goal is to identify which lines are actual axes and use them to construct subplots.

#### Key Definitions:
- **Axis**: An axis is a straight line that defines either the horizontal (x-axis) or vertical (y-axis) reference for a plot. Axes typically have ticks, which can be either numeric or string labels, but must be consistently placed along the same straight line. The detected lines in the image, labeled in blue (vertical) and red (horizontal), may or may not represent true axes. A set of ticks can only belong to one axis. No two axes can share the same set of ticks.

- **Subplot**: A subplot is a distinct plot that consists of data points organized by a pair of axes (an x-axis and a y-axis). Each subplot is characterized by the unique combination of one x-axis and one y-axis. Even if the data points in the plot are disjointed, as long as they share the same x-axis and y-axis, they form a single subplot. The number of subplots is determined by the unique axis pairings, not by the number of data points.

#### Instructions:

1. **Identify the Actual Axes**:
   - Begin by identifying which of the labeled lines are true axes.
     - Axes are defined as lines with consistent tick marks. These ticks may be numbers or strings but should lie on the same straight line.
     - Not every labeled line is an axis. Some lines are simply auxiliary straight lines and should not be used in constructing the subplots.
   - Do not infer the number of subplots from the number of labeled lines or data points. Only use the valid axes for subplot construction.
   - The indices of the lines are in random order and do not have any inherent meaning. The actual axes may not be labeled with consecutive indices and may be the ones with higher indices.

2. **Count the Subplots and Identify Their Types**:
   - A subplot is formed by a unique combination of one x-axis and one y-axis. Count the total number of unique subplot combinations by pairing valid x-axes with valid y-axes.
   - Subplot types:
     - **Dot Plot**: A subplot consisting of unconnected points.
     - **Histogram**: A subplot with rectangular bars representing data distribution.
     - **Continuous Plot**: A subplot where data points are connected by a continuous line.
   - When identifying the number and type of subplots, focus only on the axes. The data points themselves do not determine subplot count or type.

3. **Construct Subplots**:
   - For each identified subplot, create a `Subplot` object using the valid axes:
     1. **Select the Axes**: Each subplot requires one x-axis and one y-axis from the labeled lines.
     2. **Set Axis Labels**: Before using the axes, assign their tick labels using the functions `set_labels(x[i], labels)` for x-axes and `set_labels(y[i], labels)` for y-axes.
        - Tick labels should match the values on the detected axes:
          - **X-axis**: Labels should be ordered from left to right.
          - **Y-axis**: Labels should be ordered from top to bottom.
        - If ticks are strings, use the exact string values. If ticks are numbers, use the appropriate numeric values (either integer or float).
   - **Pay Attention to Axis Naming**: The axis names and their positions may not always correspond. Ensure that you use the correct axes as labeled in the chart for each subplot.

### Reminder:
- Do not rely on the position or number of data points when determining the subplots.
- The axis names and indices are crucial—verify that the selected axes are correctly identified in the provided image labels.

Let's think step-by-step. 
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
            top_p=1,
            functions=[
                {
                    "name": "DotPlot",
                    "description": "Create a DotPlot object",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "object",
                                "description": "The x-axis of the plot",
                            },
                            "y": {
                                "type": "object",
                                "description": "The y-axis of the plot",
                            },
                            "subplot_value": {
                                "type": "string",
                                "description": "The title or the dependent variable name of the subplot",
                            },
                            "has_error_bars": {
                                "type": "boolean",
                                "description": "Whether the data points in this subplot has error bars",
                            },
                            "value_direction": {
                                "type": "string",
                                "description": "The direction of the values in the plot, whether they are on the x-axis or y-axis",
                            },
                        },
                        "required": ["x", "y", "subplot_value", "has_error_bars", "value_direction"],
                    }
                },
                {
                    "name": "set_labels",
                    "description": "Set the labels of the axis based on the ticks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "axis": {
                                "type": "object",
                                "description": "The axis object to set the labels",
                            },
                            "labels": {
                                "type": "array",
                                "description": "The labels of the axis",
                                "items": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "integer"},
                                        {"type": "number"}
                                    ]
                                }
                            },
                        },
                        "required": ["labels"],
                    }
                },
            ]
        )
        print(response.choices[0].message.content)
        print(response.choices[0].message.function_call)
        print("# of input tokens: ", response.usage.prompt_tokens)
        print("# of output tokens: ", response.usage.completion_tokens)
        
        # subplots = []
        # start = response.choices[0].message.content.find("```python\n")
        # ends = response.choices[0].message.content.find("```", start+1)
        # if response.choices[0].message.content.find("```python\n", ends+1) != -1:
        #     print("Multiple code blocks detected")
        # else:
        #     exec(response.choices[0].message.content[start+9:ends])
        #     return subplots
