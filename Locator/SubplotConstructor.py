import cv2
import numpy as np
from tqdm import tqdm

MIN_LINE = 20
MIN_GAP = 5
MIN_TICK_LEN = 2
MAX_TICK_LEN = 50
MAX_ALLOWED_GAP = 5

class Axis:
    def __init__(self, line_r, line_c_lo, line_c_hi, direction, color=None):
        self.direction = direction
        self.line_r_lo = line_r
        self.line_r_hi = line_r+1
        self.line_c_lo = line_c_lo
        self.line_c_hi = line_c_hi
        self.color = color
        self.ticks = None
        self.labels = None
    
    def set_labels(self, labels):
        # Check the number of labels with the number of ticks
        pass
    
    def merge(self, axis):
        self.line_r_lo = min(self.line_r_lo, axis.line_r_lo)
        self.line_r_hi = max(self.line_r_hi, axis.line_r_hi)
        self.line_c_lo = min(self.line_c_lo, axis.line_c_lo)
        self.line_c_hi = max(self.line_c_hi, axis.line_c_hi)

class SubplotConstructor:
    def __init__(self):
        self.bg_color = np.array([255, 255, 255])
    
    def filter_axes(self, image, potential_axes, strict=False):
        axes = []
        for axis in potential_axes:
            # Find the ticks with the same color on at least one side
            sting_pos = {}
            has_lo_tick = False
            has_hi_tick = False
            for i in range(axis.line_c_lo, axis.line_c_hi):
                for j in range(axis.line_r_hi, image.shape[0]):
                    if np.array_equal(image[j][i], self.bg_color) or \
                       (j - axis.line_r_hi+1 > MIN_TICK_LEN and not np.array_equal(image[j][i], axis.color)) or \
                       (strict and not np.array_equal(image[j][i], axis.color)):
                        if MIN_TICK_LEN < j - axis.line_r_hi + 1 < MAX_TICK_LEN:
                            sting_pos[i] = 1
                            has_hi_tick = True
                        break
                for j in range(axis.line_r_lo-1, 0, -1):
                    if np.array_equal(image[j][i], self.bg_color) or \
                       (axis.line_r_lo - j - 1 > MIN_TICK_LEN and not np.array_equal(image[j][i], axis.color)) or \
                       (strict and not np.array_equal(image[j][i], axis.color)):
                        if MIN_TICK_LEN < axis.line_r_lo - j - 1 < MAX_TICK_LEN:
                            sting_pos[i] = 1
                            has_lo_tick = True
                        break
            # Group continuous ticks and represent them with the middle point
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

            # Should have at least one tick
            if len(ticks) > 0:
                axis.ticks = ticks
                if has_lo_tick and not has_hi_tick:
                    axis.line_r_hi = axis.line_r_lo + 1
                elif has_hi_tick and not has_lo_tick:
                    axis.line_r_lo = axis.line_r_hi - 1
                axes.append(axis)
        return axes

    def merge_straight_lines(self, image, potential_axes, strict=False):
        merged_lines = []
        for i in range(len(potential_axes)):
            merged = False
            for axis in merged_lines:
                # Can this line be merged with an existing axis
                if strict and not np.array_equal(axis.color, potential_axes[i].color):
                    continue
                to_merge = False
                if axis.line_c_lo <= potential_axes[i].line_c_hi and potential_axes[i].line_c_lo <= axis.line_c_hi:
                    # If the two lines overlap at the column direction
                    to_merge = True
                    for k in range(max(axis.line_c_lo, potential_axes[i].line_c_lo), min(axis.line_c_hi, potential_axes[i].line_c_hi)):
                        allowed_gap = MAX_ALLOWED_GAP
                        for l in range(min(axis.line_r_hi, potential_axes[i].line_r_hi), max(axis.line_r_lo, potential_axes[i].line_r_lo)):
                            if (strict and not np.array_equal(image[l][k], axis.color)) or \
                               (not strict and np.array_equal(image[l][k], self.bg_color)):
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
                            if (strict and not np.array_equal(image[k][l], axis.color)) or \
                               (not strict and np.array_equal(image[k][l], self.bg_color)):
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

    def estimate_my(self, image, direction, strict=False):
        if direction == "y":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # identify straight lines
        potential_axes = []
        for i in range(image.shape[0]):
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
        potential_axes = self.merge_straight_lines(image, potential_axes, strict)
        axes = self.filter_axes(image, potential_axes, strict)
        if len(axes) == 0:
            # If no axes with ticks are detected, return the straight lines
            axes = potential_axes
        
        if direction == "y":
            for axis in axes:
                axis.line_c_lo, axis.line_c_hi = image.shape[1]-axis.line_c_hi, image.shape[1]-axis.line_c_lo
                if axis.ticks is not None:
                    for i in range(len(axis.ticks)):
                        axis.ticks[i] = image.shape[1] - axis.ticks[i]
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return axes
    
    def draw_axes(self, image, axes):
        for i, axis in enumerate(axes):
            text = axis.direction + "[" + str(i) + "]"
            height, width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            if axis.direction == "x":
                color = (0, 0, 255)
                cv2.line(image, (axis.line_c_lo, axis.line_r_lo), (axis.line_c_hi, axis.line_r_lo), color, 1)
                cv2.putText(image, text, ((axis.line_c_hi + axis.line_c_lo)//2, axis.line_r_lo + width), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (255, 0, 0)
                cv2.line(image, (axis.line_r_lo, axis.line_c_lo), (axis.line_r_lo, axis.line_c_hi), color, 1)
                cv2.putText(image, text, (axis.line_r_hi, (axis.line_c_lo + axis.line_c_hi)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if axis.ticks is not None:
                for tick in axis.ticks:
                    if axis.direction == "x":
                        cv2.circle(image, (int(tick), axis.line_r_lo), 2, color, 2)
                    else:
                        cv2.circle(image, (axis.line_r_lo, int(tick)), 2, color, 2)
        return image

    def estimate_ticks(self, image, direction, strict=False):
        if direction == "y":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        potential_axes = []
        for line in lines:
            # Only want horizontal lines
            rho, theta = line[0]
            # if theta < np.pi/4 or theta > 3*np.pi/4:
            if np.abs(theta - np.pi/2) > np.pi/10:
                continue
            # Search for the line in the image
            a = np.cos(theta)
            b = np.sin(theta)
            y1 = rho / b
            y2 = (rho - image.shape[1] * a) / b
            for rho in range(int(np.floor(min(y1, y2))), min(image.shape[0], int(np.ceil(max(y1, y2)+1)))):
                cur_line_lo = 0
                cur_color = image[rho][0]
                for i in range(image.shape[1]):
                    if (strict and not np.array_equal(image[rho][i], cur_color)) or \
                       (not strict and np.array_equal(image[rho][i], self.bg_color)) or \
                        i == image.shape[1]-1:
                        if i - cur_line_lo > MIN_LINE:
                            potential_axes.append(Axis(rho, cur_line_lo, i, direction, image[rho][cur_line_lo]))
                        cur_line_lo = i
                        cur_color = image[rho][i]

        print(len(potential_axes))
        potential_axes = self.merge_straight_lines(image, potential_axes, strict)
        axes = self.filter_axes(image, potential_axes, strict)
        print(len(axes))
        if len(axes) == 0:
            # If no axes with ticks are detected, return the straight lines
            axes = potential_axes

        if direction == "y":
            for axis in axes:
                axis.line_c_lo, axis.line_c_hi = image.shape[1]-axis.line_c_hi, image.shape[1]-axis.line_c_lo
                if axis.ticks is not None:
                    for i in range(len(axis.ticks)):
                        axis.ticks[i] = image.shape[1] - axis.ticks[i]
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return axes