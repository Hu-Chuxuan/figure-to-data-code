import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import openai
from openai import OpenAI
import base64
import os
import csv


from Locator.Subplot import Subplot
from Locator.SubplotConstructor import SubplotConstructor

api_key = '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'
organization = 'org-5rFbX7p7v2H4Sk1C8xb17aea'


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt_response(base64_image, text):
  client = OpenAI(api_key=api_key, organization=organization)
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": text},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )

  return response.choices[0].message.content





class Histogram(Subplot):
    def __init__(self, x, y, subplot_value, has_error_bars, value_direction):
        super().__init__(x, y, subplot_value, has_error_bars, value_direction)

    # def estimate(self, image):
    #     raise NotImplementedError
    
    #@title find_axes()
    def find_axes(self, image):
        all_axes_image = image.copy()
        subplot = SubplotConstructor()

        # Find possible x_axis lines
        x_axes = subplot.estimate_ticks(all_axes_image, "x")
        print(f"There are total of {len(x_axes)} possible x-axis")
        subplot.draw_axes(all_axes_image, x_axes)

        # Find possible y_axis lines
        y_axes = subplot.estimate_ticks(all_axes_image, "y")
        print(f"There are total of {len(y_axes)} possible y-axis")
        subplot.draw_axes(all_axes_image, y_axes)
        cv2.imshow("all axes image", all_axes_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        time.sleep(1) # any brief amount of time seems to work
        cv2.destroyAllWindows()
        x_axis_idx = int(input('Which index is the x_axis?'))
        y_axis_idx = int(input('Which index is the y_axis?'))

        # Determine desired lines to be x and y axes
        x_axis = x_axes[x_axis_idx]
        y_axis = y_axes[y_axis_idx]


        axes_image = image.copy()

        # Draw determined x_axis
        x_x1, x_y1, x_x2, x_y2 = x_axis.line_c_lo, x_axis.line_r_lo, x_axis.line_c_hi, x_axis.line_r_lo
        # bottom_right = (x_x2, x_y2 - 3)
        cv2.line(axes_image, (x_axis.line_c_lo, x_axis.line_r_lo), (x_axis.line_c_hi, x_axis.line_r_lo), (0, 0, 255), 2)



        # Draw determined y_axis
        y_x1, y_y1, y_x2, y_y2 = y_axis.line_r_lo, y_axis.line_c_lo, y_axis.line_r_lo, y_axis.line_c_hi
        # top_left = (y_x1+3,y_y1)
        cv2.line(axes_image, (y_axis.line_r_lo, y_axis.line_c_lo), (y_axis.line_r_lo, y_axis.line_c_hi), (255, 0, 0), 2)



        cv2.imshow("axes image",axes_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return x_x1, x_y1, x_x2, x_y2, y_x1, y_y1, y_x2, y_y2
    
    def crop(self, image, y_y1, x_y2, y_x1, x_x2):
        cropped_img = image[y_y1:x_y2-3, y_x1+5:x_x2]
        cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

        return cropped_img, cropped_gray

    def detect_edges(self, image):
        edges = cv2.Canny(image, threshold1=100, threshold2=200)
        return edges
    
    def merge_contours(self, contours, distance_threshold):
        merged_contours = []
        for contour in contours:
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour is close to any existing merged contour
            merged = False
            for i, merged_contour in enumerate(merged_contours):
                mx, my, mw, mh = cv2.boundingRect(merged_contour)
                if (abs(x - mx) < distance_threshold and abs(y - my) < distance_threshold):
                    # Merge the contours
                    merged_contours[i] = np.concatenate((merged_contour, contour))
                    merged = True
                    break

            if not merged:
                # Add the contour as a new merged contour
                merged_contours.append(contour)

        return merged_contours

    def sort_contours(self, cnts):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    def find_bars(self, cropped_img, edges):
        # Find contours
        height, width = cropped_img.shape[:2]
        print(height, width)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = self.merge_contours(contours, 5)
        contours, boudningBoxes = self.sort_contours(contours)

        print("len contours:", len(contours))

        # Iterate thorugh contours and draw rectangles around contours
        edge_contour_img = cropped_img.copy()
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)

        bars = []

        for i in range(0, len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour)
            if y + h > 2*height/3 and w < (width/2):
                print(x,y,w,h, area)
                cv2.rectangle(edge_contour_img, (x, y), (x + w, y + h), (0,255,0), 2)
                bars.append((x,y,w,h))
        return bars, edge_contour_img

    def get_y_axis_info(self, full_image_path):
        prompt = '''
        You are given a picture of a plot. Your task is determine 4 things from the on the y-axis:
            1. How many tick marks with tick labels there are on the y-axis.
            2. The minimum tick label on the y-axis.
            3. The maximum tick label on the y-axis.
            4. The interval in between the first and the second tick label on y-axis.

        First, these are the definition of tick mark and label:
            Tick Marks:
            Definition: Tick marks are small lines on the axes of a plot that indicate specific intervals of measurement. They help viewers gauge values along the axes.
            Types: There are usually two types:
                Major Tick Marks: These are longer and often associated with the main divisions of the axis. They usually have labels.
                Minor Tick Marks: These are shorter and help to indicate finer divisions without labels, providing additional reference points.
            Labels:
            Definition: Labels are text descriptions placed alongside tick marks on the axes. They indicate the values or categories that the tick marks represent.
            Tick Labels: These are the numerical or categorical values shown next to the tick marks. They provide the viewer with a clear understanding of the scale and the data represented.

        What you must do to get the number of tick labels is:
            1. identify tick marks with tick labels next to it.
            2. Count how many tick marks with tick labels next to it.
            3. If the last tick label is below the end of the plot, add one more to the count
            4. Give back the answer with only the number

        What you must do to get minimum tick label is:
            1. identify tick marks with tick labels next to it.
            2. Identify the minimum tick labels on the y-axis.
            3. Give back the answer with only the number

        What you must do to get maximum tick label is:
            1. identify tick marks with tick labels next to it.
            2. Identify the maximum tick labels on the y-axis.
            3. Give back the answer with only the number

        What you must do to get the interval between the first and second tick labels is:
            1. identify tick marks with tick labels next to it.
            2. Identify the first and second tick labels on the y-axis.
            3. Subtract the first label from the second label to get the interval.
            4. Give back the answer with only the number

        Before you give me the answer check 3 times to make sure the answer is correct. Make sure you are only counting the tick marks on the y-axis.
        Give the answer in this format: num_y_ticks,min_y_labels,max_y_labels,interval_between_y_labels
        '''

        base64_image = encode_image(full_image_path)

        answer = get_gpt_response(base64_image, prompt)
        splitted = answer.split(",")
        num_y_ticks, min_y_label, max_y_label, y_interval = splitted
        return float(num_y_ticks), float(min_y_label), float(max_y_label), float(y_interval)

    def get_x_axis_info(self, full_image_path):
        prompt = '''
        You are given a picture of a plot. Your task is determine 4 things from the on the x-axis:
            1. How many tick marks with tick labels there are on the x-axis.
            2. The minimum tick label on the x-axis.
            3. The maximum tick label on the x-axis.
            4. The interval in between the first and the second tick label on x_axis.

        First, these are the definition of tick mark and label:
            Tick Marks:
            Definition: Tick marks are small lines on the axes of a plot that indicate specific intervals of measurement. They help viewers gauge values along the axes.
            Types: There are usually two types:
                Major Tick Marks: These are longer and often associated with the main divisions of the axis. They usually have labels.
                Minor Tick Marks: These are shorter and help to indicate finer divisions without labels, providing additional reference points.
            Labels:
            Definition: Labels are text descriptions placed alongside tick marks on the axes. They indicate the values or categories that the tick marks represent.
            Tick Labels: These are the numerical or categorical values shown next to the tick marks. They provide the viewer with a clear understanding of the scale and the data represented.

        What you must do to get the number of tick labels on x-axis is:
            1. identify tick marks with tick labels next to it on x-axis.
            2. Count how many tick marks with tick labels next to it.
            3. Give back the answer with only the number

        What you must do to get minimum tick label on x-axis is:
            1. identify tick marks with tick labels next to it on x-axis.
            2. Identify the minimum tick labels on the x-axis.
            3. Give back the answer with only the number

        What you must do to get maximum tick label on x-axis is:
            1. identify tick marks with tick labels next to it on x-axis.
            2. Identify the maximum tick labels on the x-axis.
            3. Give back the answer with only the number

        What you must do to get the interval between the first and second tick labels on x-axis is:
            1. identify tick marks with tick labels next to it on x-axis.
            2. Identify the first and second tick labels on the x-axis.
            3. Subtract the first label from the second label to get the interval.
            4. Give back the answer with only the number

        Before you give me the answer check 3 times to make sure the answer is correct. Make sure you are only counting the tick marks on the x-axis.
        Give the answer in this format: num_y_ticks,min_y_labels,max_y_labels,interval_between_y_labels
        '''

        base64_image = encode_image(full_image_path)

        answer = get_gpt_response(base64_image, prompt)
        splitted = answer.split(",")
        num_x_ticks, min_x_label, max_x_label, x_interval = splitted
        return float(num_x_ticks), float(min_x_label), float(max_x_label), float(x_interval)

    def get_x_axis_labels(self, full_image_path):
        prompt = '''
        You are given a picture of a plot. Your task is determine the tick labels on the x-axis.

        First, these are the definition of tick mark and label:
            Tick Marks:
            Definition: Tick marks are small lines on the axes of a plot that indicate specific intervals of measurement. They help viewers gauge values along the axes.
            Types: There are usually two types:
                Major Tick Marks: These are longer and often associated with the main divisions of the axis. They usually have labels.
                Minor Tick Marks: These are shorter and help to indicate finer divisions without labels, providing additional reference points.
            Labels:
            Definition: Labels are text descriptions placed alongside tick marks on the axes. They indicate the values or categories that the tick marks represent.
            Tick Labels: These are the numerical or categorical values shown next to the tick marks. They provide the viewer with a clear understanding of the scale and the data represented.

        What you must do is:
            1. identify tick marks with tick labels next to it on the x_axis.
            2. Identify what the tick labels says.
            3. Give back the answer in a comma seperated format with no spaces in between.

        Before you give me the answer check 3 times to make sure the answer is correct. Make sure you are only counting the tick marks on the x-axis.
        Make sure to give the labels in a comma separated format with no spaces in between. Only give me the list of tick labels, nothing else.
        '''

        base64_image = encode_image(full_image_path)

        answer = get_gpt_response(base64_image, prompt)
        labels = answer.split(",")

        print("x_labels: ", labels)
        return labels

    def construct_bar_plot(self, cropped_img, bars, image_path):
        num_y_ticks, min_y_label, max_y_label, interval = self.get_y_axis_info(image_path)
        # num_y_ticks = 6
        print(num_y_ticks)
        cropped_img_height = cropped_img.shape[0]
        print(cropped_img_height)
        # interval = 25
        unit_to_pxl_ratio = interval/(cropped_img_height/(max_y_label/interval))
        heights = []

        # Number of bars
        x_labels = self.get_x_axis_labels(image_path)
        # x_labels = list(range(len(bars)))

        for bar in bars:
            print(bar)
            x,y,w,h = bar
            heights.append(unit_to_pxl_ratio * (h+5))

        print(heights)

        # Export to CSV
        with open(image_path.split(".")[0] + ".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type-1', 'Value'])  # Write header
            writer.writerows(zip(x_labels, heights))  # Write data rows
            # writer.writerow([f'{bin_edges[i]} - {bin_edges[i] + bar_widths[i]}', f'{bar_heights[i]}'])  # Write each bin start and height
        
        print("finished writing bar csv")

        # # Plotting
        # plt.figure(figsize=(8, 6))
        # plt.bar(x_labels, heights, color='olive')
        # plt.xlabel("x label")
        # plt.ylabel("y label")
        # plt.title("Title")
        # plt.show()

        print("finished showing bar")

    
    def construct_histogram(self, cropped_img, bars, image_path):
        num_y_ticks, min_y_label, max_y_label, y_interval = self.get_y_axis_info(image_path)
        print(num_y_ticks, min_y_label, max_y_label, y_interval)
        cropped_img_height = cropped_img.shape[0]
        y_unit_to_pxl_ratio = y_interval/(cropped_img_height/(max_y_label/y_interval))
        # y_unit_to_pxl_ratio = 25/(cropped_img_height/(4))

        num_x_ticks, min_x_label, max_x_label, x_interval = self.get_x_axis_info(image_path)
        print(num_x_ticks, min_x_label, max_x_label, x_interval)
        cropped_img_width = cropped_img.shape[1]
        x_unit_to_pxl_ratio = x_interval/(cropped_img_height/(max_x_label/x_interval))
        # x_unit_to_pxl_ratio = 0.1/(cropped_img_height/10)

        # Extract x positions, widths, and heights from each bar
        bar_x_positions = [bar[0] for bar in bars]   # x positions of each bar
        bar_heights = [bar[3] for bar in bars]       # heights of each bar (y-axis)
        bar_widths = [bar[2] for bar in bars]        # widths of each bar (x-axis)

        # Sort bars by x position to maintain order
        sorted_indices = sorted(range(len(bar_x_positions)), key=lambda i: bar_x_positions[i])
        bar_x_positions = [bar_x_positions[i]* x_unit_to_pxl_ratio for i in sorted_indices]
        bar_heights = [bar_heights[i] * y_unit_to_pxl_ratio for i in sorted_indices]
        bar_widths = [bar_widths[i] * x_unit_to_pxl_ratio for i in sorted_indices]

        # Construct bin edges based on x positions and widths
        bin_edges = [x for x in bar_x_positions]
        bin_edges.append(bin_edges[-1] + bar_widths[-1])  # Last bin edge

        print("bar_x_positions: ", bar_x_positions)
        print("bar_heights: ", bar_heights)
        print("bar_widths: ", bar_widths)
        print("bin_edges: ", bin_edges)

        # Plot the histogram using bin edges and heights as weights
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=bar_heights, align='left', rwidth=1)
        plt.xlabel('X-axis (Scaled to Original)')
        plt.ylabel('Frequency (Scaled to Original)')
        plt.title('Histogram Reconstructed from Contours')
        plt.show()

        # Export to CSV
        with open(image_path.split(".")[0] + ".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type-1', 'Value'])  # Write header
            for i in range(len(bar_heights)):
                writer.writerow([f'{bin_edges[i]} - {bin_edges[i] + bar_widths[i]}', f'{bar_heights[i]}'])  # Write each bin start and height

    def estimate(self, image_path):
        image = cv2.imread(image_path)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        x_x1, x_y1, x_x2, x_y2, y_x1, y_y1, y_x2, y_y2 = self.find_axes(image)
        cropped_img, cropped_gray = self.crop(image, y_y1, x_y2, y_x1, x_x2)
        cv2.imshow("cropped gray", cropped_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        edges = self.detect_edges(cropped_gray)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        bars, edge_contour_img = self.find_bars(cropped_img, edges)
        cv2.imshow("edge_contour_img", edge_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # hist_or_bar = histogram_or_bar(image_path)
        hist_or_bar = input("'Histogram' or 'Bar'?")
        if hist_or_bar == 'Histogram':
          self.construct_histogram(cropped_img, bars, image_path)
        else:
          self.construct_bar_plot(cropped_img, bars, image_path)
