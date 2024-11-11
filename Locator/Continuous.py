import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
from PIL import Image
from scipy import interpolate
import openai
from openai import OpenAI

from scipy.spatial import KDTree
from collections import defaultdict









def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def detect_rectangle_in_any_quarter(image_path):
    # Load the image and convert it to grayscale
    src = cv2.imread(image_path)
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to improve contrast in dark areas
    gray_image = clahe(gray_image)
    
    # Apply adaptive thresholding (inverted to highlight white areas)
    img_th = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 2)
    
    # Define right quarter region limits
    height, width = gray_image.shape
    quarter_width, quarter_height = width // 2, height // 2
    
    # Coordinates for only the right quarters
    right_quarters = {
        "top_right": (quarter_width, 0, width, quarter_height),
        "bottom_right": (quarter_width, quarter_height, width, height)
    }
    
    # Find contours
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare to find the largest contour within the right quarters
    max_size = 0
    largest_rectangle = None
    canvas = src.copy()
    
    # Process each contour
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        area = w * h
        
        # Check if the contour meets the size threshold
        if area > 500:  # Reduced threshold to consider smaller areas
            # Check if the rectangle is contained within any of the right quarters
            for (qx1, qy1, qx2, qy2) in right_quarters.values():
                if qx1 <= x < qx2 and qy1 <= y < qy2 and qx1 <= x + w <= qx2 and qy1 <= y + h <= qy2:
                    # If it meets all criteria, update the largest rectangle if necessary
                    if area > max_size:
                        largest_rectangle = rect
                        max_size = area
                    break

    # Create and display the masked image with only the largest rectangle
    if largest_rectangle is not None:
        x, y, w, h = largest_rectangle
        
        # Initialize a black canvas with the same size as the source image
        mask = np.zeros_like(src)
        
        # Copy the contents of the largest rectangle from the original image to the mask
        mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
        
        # Display the masked image with only the largest rectangle on a black background
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Masked Image with Only the Largest Rectangle")
        plt.show()
        
        # Save the mask if needed
        # cv2.imwrite("masked_output.png", mask)
        
        print("Detected rectangle at:", x, y, w, h)
        return x, y, w, h, mask
    else:
        print("No suitable contour found within the right quarters.")
        return None




def read_image(image_path):
  return Image.open(image_path)

def plt_plot(img, title="Image"):
  plt.figure(figsize=(6, 6))
  plt.imshow(img)
  plt.title(title)
  plt.show()

def plt_plot_scatter(x, y, title="Scatter Plot"):
    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, color='black', s=1)

    plt.title("Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()






def get_coordinates(x_min, x_max, y_min, y_max, dist_to_origin_x, dist_to_origin_y, dist_to_end_x, dist_to_end_y, x_ratio_denom, y_ratio_denom):

    ratio_x = (x_max - x_min) / x_ratio_denom
    ratio_y = (y_max - y_min) / y_ratio_denom
    x_min = x_min - ratio_x*dist_to_origin_x
    x_max = x_max + ratio_x*dist_to_end_x


    y_min = y_min - ratio_y*dist_to_origin_y
    y_max = y_max + ratio_y*dist_to_end_y

    print("got_coordinates: ", x_min, x_max, y_min, y_max)
    return x_min, x_max, y_min, y_max

def crop_image(x, y, width, height):
    return image[y:y+height, x:x+width]


def get_pixels(image, contour):
    result = cv2.drawContours(image.copy(), contour, -1, (0, 255, 0), -1)
    x = []
    y = []

    for i in range(result.shape[1]):
        for j in range(result.shape[0]):
            if result[j][i][0] == 0 and result[j][i][1] == 255 and result[j][i][2] == 0:
                x.append(i)
                y.append(j)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, result.shape[0] - np.array(y), color='red', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return x, y

def scale_to_original(og_min, og_max, new_min, new_max, col):
    scaled_col = []
    for value in col:
        normalized_value = (value - new_min) / (new_max - new_min)
        og = normalized_value * (og_max - og_min) + og_min
        scaled_col.append(og)
    return scaled_col


def average_ys(xs, ys):
    df = pd.DataFrame({'x': xs, 'y': ys})
    df_avg = df.groupby('x')['y'].mean().reset_index()
    x_avg = df_avg['x'].to_list()
    y_avg = df_avg['y'].to_list()
    return x_avg, y_avg

def interpolate_data(x_avg, y_avg):
    x_new_gray = np.linspace(min(x_avg), max(x_avg), num=10000)  # 500 points for a smooth curve
    f_interpolated = interpolate.interp1d(x_avg, y_avg, kind='linear')
    y_interpolated_gray = f_interpolated(x_new_gray)
    return x_new_gray, y_interpolated_gray


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def get_gpt_response(base64_image, text):
  client = OpenAI(api_key = "secret")
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

def get_axes_bounds(full_image_path):

  base64_image = encode_image(full_image_path)

  x_min = float(get_gpt_response(base64_image, "What is the smallest value label on the x axis? Respond with only the number").replace(',', ''))
  y_min = float(get_gpt_response(base64_image, "What is the smallest value label on the y axis? Respond with only the number").replace(',', ''))
  x_max = float(get_gpt_response(base64_image, "What is the largest value label on the x axis? Respond with only the number").replace(',', ''))
  y_max = float(get_gpt_response(base64_image, "What is the largest value label on the y axis? Respond with only the number").replace(',', ''))
  print(x_min, x_max, y_min, y_max)
  return x_min, x_max, y_min, y_max


def there_is_a_legend(image_path):
    base64_image = encode_image(image_path)
    answer = get_gpt_response(base64_image, "is there a legend in this image? If there is, respond with 'yes'. If not, respond with 'no'.").replace(".", '').lower()
    if answer == "yes":
        return True
    return False




def get_clusters(image):

    to_return = []
    image = image.convert("RGB")

    image_data = np.array(image)

    # Flatten image data for easier processing
    pixels = image_data.reshape(-1, 3)
    pixel_coords = [(i // image.width, i % image.width) for i in range(len(pixels))]

    # Define RGB distance threshold
    threshold = 50  
    white_threshold = 240  


    tree = KDTree(pixels)


    clusters = defaultdict(list)
    visited = set()

    # Group pixels by RGB similarity
    for idx, pixel in enumerate(pixels):
        if idx not in visited:
            # Find all points within the threshold distance
            indices = tree.query_ball_point(pixel, threshold)
            clusters[tuple(pixel)].extend([(pixels[i], pixel_coords[i]) for i in indices])
            visited.update(indices)

    # Identify clusters that contain white or near-white pixels
    white_clusters = set()
    for color, pixel_info in clusters.items():
        if any(np.all(np.array(pixel) >= white_threshold) for pixel, _ in pixel_info):
            white_clusters.update(coord for _, coord in pixel_info)

    # Remove white-clustered pixels from all other clusters
    for color in list(clusters.keys()):
        clusters[color] = [(pixel, coord) for pixel, coord in clusters[color] if coord not in white_clusters]

    # Post-process clusters to remove overlapping coordinates if overlap > 20%
    filtered_clusters = {}
    assigned_coords = {}

    for color, coords in clusters.items():
        # Convert coords to a set of pixel coordinates
        coord_set = set(coord for _, coord in coords)

        # Check if any coordinates overlap with previously assigned clusters
        overlapping_coords = coord_set.intersection(assigned_coords.keys())
        overlap_percentage = len(overlapping_coords) / len(coord_set) if coord_set else 0

        if overlapping_coords and overlap_percentage > 0.8:
            # Calculate the size of current cluster and all overlapping clusters
            cluster_size = len(coord_set)
            overlap_size = sum(assigned_coords[coord] for coord in overlapping_coords)

            # Only keep the current cluster if it is larger than the overlapping clusters
            if cluster_size > overlap_size:
                # Remove the coordinates from smaller clusters
                for coord in overlapping_coords:
                    del assigned_coords[coord]
                # Add this cluster's coordinates to the assigned list
                filtered_clusters[color] = coords
                for coord in coord_set:
                    assigned_coords[coord] = cluster_size
        else:
            # No overlap or overlap is below threshold, so add this cluster to filtered_clusters
            filtered_clusters[color] = coords
            for coord in coord_set:
                assigned_coords[coord] = len(coords)

    # yellow here is actually white sorry for the confusion
    # Define yellow color for output
    yellow = [255, 255, 255]  # Slight change for visibility
    total_non_white_pixels = sum(len(coords) for coords in filtered_clusters.values())
    non_white_threshold = 0.05 * total_non_white_pixels

    # Process each cluster for visualization and yellowing of columns
    for i, (color, coords) in enumerate(filtered_clusters.items()):
        # Start with a new white background image for each cluster
        output_image = np.ones_like(image_data) * 255  # All white initially
        
        # Track non-white pixels in each column
        column_pixels = defaultdict(list)
        
        # Populate the cluster pixels into the new image and track column-wise pixels
        for _, (row, col) in coords:
            output_image[row, col] = color
            column_pixels[col].append(row)
        
        # Iterate through columns in this cluster to adjust based on highest and lowest rows
        for col, rows in column_pixels.items():
            if not rows:
                continue

            if len(rows) == 1:
                # Set the single pixel to yellow
                output_image[rows[0], col] = yellow

            min_row, max_row = min(rows), max(rows)
            # Check for any white pixels in the range [min_row, max_row]
            contains_white = any(np.all(output_image[r, col] >= white_threshold) for r in range(min_row, max_row + 1))
            
            # If any white pixel is within the range, set entire column to yellow
            if contains_white:
                for r in range(min_row, max_row + 1):
                    output_image[r, col] = yellow

        non_white_pixel_count = np.sum(np.all(output_image != yellow, axis=-1))
        
        # Skip plotting/saving if the image is all white or has less than 10% of total non-white pixels
        if non_white_pixel_count == 0 or non_white_pixel_count < non_white_threshold:
            continue  
        
        to_return.append(output_image)

        output_image_pil = Image.fromarray(output_image.astype(np.uint8))
        output_image_pil.show()  
        # output_image_pil.save(f"cluster_{i}_visualization.png")


    return to_return



# get axis version 2

def get_axis(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to make black lines prominent
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Use Canny edge detection to detect edges
    edges = cv2.Canny(binary, 50, 150)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=0, minLineLength=1, maxLineGap=0)




    # Create a copy of the original image to draw the lines
    image_with_lines = image.copy()

    # Draw each line on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line for visibility

    # Display the image with detected lines
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.axis('off')  # Hide axis numbers
    plt.title("Lines Detected with Hough Transform")
    plt.show()



    
    # Initialize list to store detected horizontal axis lines
    axis_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check if line is horizontal (small angle)
            if abs(y1 - y2) < 5:  # Adjust tolerance as needed
                axis_lines.append((x1, y1, x2, y2))
    
    def has_white_pixel_between(image, x1, y1, x2, y2):
        # Ensure x1 <= x2 for consistent traversal
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        # Calculate the number of points to sample along the line
        num_points = int(np.hypot(x2 - x1, y2 - y1))
        
        # Check each sampled point along the line
        for i in range(num_points):
            x = int(x1 + i * (x2 - x1) / num_points)
            y = int(y1 + i * (y2 - y1) / num_points)
            
            # Check if the pixel at (x, y) is white in the binary image
            if image[y, x] > 240:  # Assuming white is any value > 200
                return True
        return False
    
    # Group lines by their vertical position to find clusters of horizontal lines
    axis_clusters = []
    for line in axis_lines:
        x1, y1, x2, y2 = line
        added_to_cluster = False
        for cluster in axis_clusters:
            # Check if this line is close in vertical position to an existing cluster


            cluster_y_values = [ (line[1] + line[3]) / 2 for line in cluster ]
            cluster_mean_y = np.mean(cluster_y_values)

            if abs(cluster_mean_y - y1) < 5:  # Adjust tolerance as needed
                # Check for white pixels between the last line in the cluster and this line
                last_x1, last_y1, last_x2, last_y2 = cluster[-1]
                if not has_white_pixel_between(binary, last_x1, cluster_mean_y, x1, cluster_mean_y) and not has_white_pixel_between(binary, last_x1, last_y2, x1, y2) and not has_white_pixel_between(binary, last_x1, last_y1, x1, y1):
                    cluster.append(line)
                    added_to_cluster = True
                    #break
        if not added_to_cluster:
            axis_clusters.append([line])
    
    # Choose the most prominent horizontal line cluster as the axis
    max_combined_x_distance = 0
    horizontal_axis = None

    for cluster in axis_clusters:
        # Calculate the combined x distance for the cluster
        

        # if len(cluster) > 15:
        #     cluster_image = image.copy()
            
        #     # Draw each line in the cluster
        #     for line in cluster:
        #         x1, y1, x2, y2 = line
        #         cv2.line(cluster_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red color for cluster lines
            
        #     # Display the image for the current cluster
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(cv2.cvtColor(cluster_image, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        #     plt.title(f"Cluster with {len(cluster)} lines")
        #     plt.show()


        combined_x_distance = sum(line[2] - line[0] for line in cluster)  # Sum of x2 - x1 for each line in the cluster
        
        # Choose the cluster with the maximum combined x distance
        if combined_x_distance > max_combined_x_distance:
            max_combined_x_distance = combined_x_distance
            horizontal_axis = cluster

    #print("horizontal_axis", horizontal_axis)
    
    # Draw the detected axis line on the image for visualization
    if horizontal_axis:
        cluster_y_values = [ (line[1] + line[3]) / 2 for line in horizontal_axis ]
        axis_row = int(-(np.mean(cluster_y_values) // -1))
        #axis_row = min(line[1] for line in horizontal_axis)
        min_col = min(line[0] for line in horizontal_axis)
        max_col = max(line[2] for line in horizontal_axis)
        cv2.line(image, (min_col, axis_row), (max_col, axis_row), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    return gray, axis_row, min_col, max_col


















# def get_axis(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to make black lines prominent
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Use Canny edge detection to detect edges
    edges = cv2.Canny(binary, 50, 150)
    
    # Detect lines using Hough Line Transform
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # Initialize list to store detected horizontal axis lines
    axis_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check if line is horizontal (small angle)
            if abs(y1 - y2) < 5:  # Adjust tolerance as needed
                axis_lines.append((x1, y1, x2, y2))
    
    # Group lines by their vertical position to find clusters of horizontal lines
    axis_clusters = []
    for line in axis_lines:
        x1, y1, x2, y2 = line
        added_to_cluster = False
        for cluster in axis_clusters:
            # Check if this line is close in vertical position to an existing cluster
            if abs(cluster[0][1] - y1) < 5:  # Adjust tolerance as needed
                cluster.append(line)
                added_to_cluster = True
                break
        if not added_to_cluster:
            axis_clusters.append([line])
    
    # Choose the most prominent horizontal line cluster as the axis
    max_span = 0
    horizontal_axis = None
    for cluster in axis_clusters:
        min_col = min(line[0] for line in cluster)
        max_col = max(line[2] for line in cluster)
        span = max_col - min_col  # Calculate the span of the cluster

        if span > max_span:
            max_span = span
            horizontal_axis = cluster

    print("horizontal_axis", horizontal_axis)
    
    # Determine the row of the axis and its min and max column positions
    axis_row = min(line[1] for line in horizontal_axis)
    min_col = min(line[0] for line in horizontal_axis)
    max_col = max(line[2] for line in horizontal_axis)

    print(axis_row, min_col, max_col)
    
    cv2.line(image, (min_col, axis_row), (max_col, axis_row), (0, 255, 0), 2)
    
    # Detect ticks along the axis
    #result_image, tick_positions = detect_vertical_ticks(image, binary, axis_row, min_col, max_col)
    
    return gray, axis_row, min_col, max_col  #, tick_positions

def get_ticks(gray, axis_row, min_col, max_col, skip_distance=10, black_threshold=180):
    ticks = [] # may need to chnage this for rotated image !!!!!!!!!
    col = min_col
    
    while col <= max_col:
        is_tick = True
        for row_offset in range(1, 9):
            pixel_value = gray[axis_row + row_offset, col]
            #print(pixel_value)
            if pixel_value > black_threshold:
                #print("broke")
                is_tick = False
                break
        
        if is_tick:
            ticks.append((axis_row, col))
            col += skip_distance
        else:
            col += 1
    
    return ticks









def get_non_white_pixels(classified_color):
    x = []
    y = []

    for i in range(len(classified_color)):
        for j in range(len(classified_color[0])):
            if classified_color[i][j][0] > 240 and classified_color[i][j][1] > 240 and classified_color[i][j][2] > 240:
                continue
            else:
                x.append(j)
                y.append(i)

    return x, y



def transform_single_column_to_row(col_prime, image_width):
    # Step 1: Reverse the horizontal flip
    col_double_prime = image_width - col_prime - 1
    
    # Step 2: Reverse the 90-degree counterclockwise rotation to get the original row
    row_original = col_double_prime
    
    return row_original

# Function to transform points back to the original coordinates
def transform_back_to_original(points, image_height, image_width):
    original_points = []
    for (row_prime, col_prime) in points:
        # Step 1: Reverse the horizontal flip
        col_double_prime = image_width - col_prime - 1
        
        # Step 2: Reverse the 90-degree counterclockwise rotation
        row_original = col_double_prime
        col_original = image_height - row_prime - 1
        
        original_points.append((row_original, col_original))
    
    return original_points


def plot_ticks_on_image(image, ticks):
    for tick in ticks:
        # Draw a blue dot (filled circle) at each tick position
        axis_row, col = tick
        cv2.circle(image, (col, axis_row), radius=3, color=(255, 0, 0), thickness=-1)  # Blue dot with radius 3
    return image


def get_ratios(image_path):

    image = cv2.imread(image_path)

    gray, axis_row_x, min_col_x, max_col_x = get_axis(cv2.imread(image_path))
    x_ticks = get_ticks(gray, axis_row_x, min_col_x, max_col_x)
    result_image = plot_ticks_on_image(image, x_ticks)

    # Display the result in Jupyter Notebook
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
    plt.axis('off')
    plt.title('Ticks Marked on Image')
    plt.show()
    print("x_ticks", x_ticks)

    rotated_image_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    correct_format = cv2.flip(rotated_image_counterclockwise, 1)
    gray_transformed, axis_row_y, min_col_y, max_col_y = get_axis(correct_format)
    y_ticks = get_ticks(gray_transformed, axis_row_y, min_col_y, max_col_y)


    result_image = plot_ticks_on_image(correct_format, y_ticks)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
    plt.axis('off')
    plt.title('Ticks Marked on Image')
    plt.show()
    print("y_ticks_inital", y_ticks)

    H, W = gray_transformed.shape[:2]
    y_ticks = transform_back_to_original(y_ticks, H, W)
    print("y_ticks", y_ticks)
    min_col_y = transform_single_column_to_row(min_col_y, W)
    max_col_y = transform_single_column_to_row(max_col_y, W)
    print("min_col_y: ", min_col_y)
    print("max_col_y: ", max_col_y)


    x_ratio_denom = x_ticks[-1][1] - x_ticks[0][1]
    dist_to_origin_x = x_ticks[0][1] - min_col_x
    dist_to_end_x = max_col_x - x_ticks[-1][1]
    origin_x = min_col_x + 3
    end_x = max_col_x
    print(x_ratio_denom, dist_to_origin_x, dist_to_end_x, origin_x, end_x)



    y_ratio_denom = y_ticks[0][0] - y_ticks[-1][0]
    dist_to_origin_y = min_col_y - y_ticks[0][0]
    dist_to_end_y = y_ticks[-1][0] - max_col_y
    origin_y = min_col_y - 3
    end_y = max_col_y
    print(y_ratio_denom, dist_to_origin_y, dist_to_end_y, origin_y, end_y)



    return dist_to_origin_x, dist_to_origin_y, dist_to_end_x, dist_to_end_y, x_ratio_denom, y_ratio_denom, origin_x, origin_y, end_x, end_y





# Usage

image_path_real = "p.png"

image_copied = cv2.imread(image_path_real).copy()
cv2.imwrite("copied_image.png", image_copied)

image_path = "copied_image.png"






remove_legend = there_is_a_legend(image_path)
print(remove_legend)
if remove_legend:
    legend_x, legend_y, legend_w, legend_h, legend_mask,  = detect_rectangle_in_any_quarter(image_path)
    cv2.imwrite("temp.png", legend_mask)
    remove_legend_2 = there_is_a_legend("temp.png")
    if remove_legend_2:
        print("legend exists????")
        image_copied[legend_y:legend_y+legend_h, legend_x:legend_x+legend_w] = (255, 255, 255)
        cv2.imwrite(image_path, image_copied)
        



image = Image.open(image_path).convert("RGB")
x_min, x_max, y_min, y_max = get_axes_bounds(image_path)
dist_to_origin_x, dist_to_origin_y, dist_to_end_x, dist_to_end_y, x_ratio_denom, y_ratio_denom, origin_x, origin_y, end_x, end_y = get_ratios(image_path)

#cropped_image = crop_image(142, 0, image.width, 1205)
x_min, x_max, y_min, y_max = get_coordinates(x_min, x_max, y_min, y_max, dist_to_origin_x, dist_to_origin_y, dist_to_end_x, dist_to_end_y, x_ratio_denom, y_ratio_denom)
print(x_min, x_max, y_min, y_max)
cropped_image = image.crop((origin_x, end_y, end_x, origin_y))



scaled_xs = []
scaled_ys = []
interpolated_xs = []
interpolated_ys = []
classified_colors = get_clusters(cropped_image)
for i in range(len(classified_colors)):
    print("inside")

    x, y = get_non_white_pixels(classified_colors[i])

    # scale to original
    scaled_xs.append(scale_to_original(x_min, x_max, 0, cropped_image.width, x))
    y = cropped_image.height - np.array(y)
    scaled_ys.append(scale_to_original(y_min, y_max, 0, cropped_image.height, y))
    #plt_plot_scatter(scaled_xs[i], scaled_ys[i], title="Scatter Plot of Edge Coordinates Original Scale")

    # average ys
    x_avg, y_avg = average_ys(scaled_xs[i], scaled_ys[i])
    if len(x_avg) < 20:
        continue
    #plt_plot_scatter(x_avg, y_avg, title="Scatter Plot of Edge Coordinates Average")

    # interpolate
    x_interpolated, y_interpolated = interpolate_data(x_avg, y_avg)
    interpolated_xs.append(x_interpolated)
    interpolated_ys.append(y_interpolated)
    #plt_plot_scatter(x_interpolated, y_interpolated, title="Scatter Plot of Edge Coordinates Interpolated")



# Initialize color map
colors = plt.cm.get_cmap("tab10")  # Use "tab10" colormap which has 10 distinct colors

# Create a new figure for the combined plot
plt.figure(figsize=(10, 6))

for i in range(len(interpolated_xs)):
    # Generate a unique color for each line based on the index
    color = colors(i % 10)  # Loop through colors if more than 10 data sets
    
    # Plot each set of interpolated_x and interpolated_y with a unique color
    plt.plot(interpolated_xs[i], interpolated_ys[i], label=f"Line {i+1}", color=color)

# Set the axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add legend, labels, and title
plt.legend()
plt.xlabel("Interpolated X")
plt.ylabel("Interpolated Y")
plt.title("Combined Plot of Interpolated Data Points with Different Colors")

# Show the combined plot
plt.show()








