import cv2
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import argparse
import pandas as pd
from io import StringIO

digitization = '''
You are given a picture of a plot or a table. Your task is to digitize the data from the picture and convert it into a CSV file. This involves extracting the data points, labels, and other relevant information from the picture and organizing them into a structured dataset. The goal is to create a digital representation of the data that can be easily analyzed, manipulated, and visualized by a computer program.

To do this, you will first identify whether the picture contains a plot or a table. 

If the picture contains a table, 
    1. In the headers, you only need to preserve the minimum amount of information to identify the columns and rows. 
    2. If a cell contains multiple statistics, you need to split them into separate **columns**. The column of the coefficient should be the same as the original name, while the rest should append the indices "(1)", "(2)", etc. Since we are generating a CSV file, you should use separate COLUMNs to represent the statistics INSTEAD OF ROWs. Therefore, you shoud first count the maximum number of statistics in a column and then determine the number of columns you need to split for each column.
    3. If the statistics are wrapped in parentheses, you should remove the parentheses and keep the statistics as they are.
    4. If a cell contains a statistic significance marker, you should remain the marker as it is. DO NOT convert the statistic significance to p-values. The marker should be attached in the same cell as the statistic it refers to.
    5. For all special characters, you should represent them as the LaTeX notation. You MUST NOT separate the statistic significance marker with the coefficients. 
    6. You MUST reserve the structure of the table in the image, which means that you CANNOT transpose the table or rearrange the orders. You can only split the table into multiple CSV files if there are multiple panels with different headers. Also, all cells MUST be quoted. 

If the picture contains a plot, 
    You output should follow the following format:
        1. You should use "Type-{}" as the column name for the independent variables. For example, in a plot with independent axis labels and multiple curves, you should use "Type-1" to represent the independent axis labels and "Type-2" for the curve labels. If the plot has only one curve or only one x-axis label, you should use "Type-1" to refer to the one independent variable.
        2. You should use "Value" as the column name for the dependent variable and "Subplot Value" as the column names for the different subplots.
        3. If the plot has error bars, you should include "Error Bar Length" as the column name for the error values. 
    To estimate the data points in a quantitative way, you shoud use pixel coordinates to calculate the data points. 
        1. First, you should identify the pixel positions of the ticks on the axis. With these pixel positions, you can have a reference point and the scale of the axis to convert the pixel positions to the actual values.
        2. Then, for each data point, you should estimate the pixel position of the mean and the diameter of the error bars in pixel. 
        3. Finally, you should calculate the mean and the diameter of the error bars to convert them in the same unit as the axis based on the pixel positions.
        4. For example, in the plot, in y-axis, 0 is about 450 pixel, 0.05 is about 350 pixel. We can use the 0 as the reference point and the scale is 100 pixel for 0.05 unit. The pixel posititon of the mean of a data point is about (100, 420), the diameter of its error bar is about 40 pixels. Consider that the 0 in y-axis is in about 450 pixel and 0.05 is in about 350 pixel, the mean should be around 0 + (450 - 420) * (0.05 - 0) / 100 = 0.015 and the diameter should be 0.05 * 40 / 100 = 0.02".
    For the dot plots or histograms, you should estimate all the data points in the plot. For the continuous plots, you should sample at least 20 points to estimate the curve. You MUST NOT omit any data points. ALL data points MUST be explicitly included in the CSV file.

You MUST use "```csv" and "```" to enclose the CSV-formatted data.

Let's think step by step. 
'''

def encode_image(image):
    # use base64 to encode the cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def digitize(image, api_key=None, organization=None):
    client = OpenAI(api_key=api_key, organization=organization)
    encoded_img = encode_image(image)
    msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": digitization,
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
        model="gpt-4o",
        messages=msg,
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
    )
    print(response.choices[0].message.content)
    print(response.choices[0].message.function_call)
    print("# of input tokens: ", response.usage.prompt_tokens)
    print("# of output tokens: ", response.usage.completion_tokens)

    response = response.choices[0].message.content
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
    for i, res in enumerate(res):
        if len(res) == 1:
            res.to_csv(args.output, index=False)
        else:
            res.to_csv(f"{args.output[:-4]}-{i}.csv", index=False)
    with open(f"{args.output[:-4]}.txt", "w") as f:
        f.write(response)

