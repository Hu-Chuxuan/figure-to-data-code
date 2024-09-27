class Curve:
    def __init__(self, color, shape, x, y, curve_value, error_bars=None):
        # Must contain the color and shape of the curve to distinguish it from other curves
        self.color = color
        self.shape = shape
        self.x = x
        self.y = y
        self.error_bars = error_bars
        self.curve_value = curve_value

class DotPlot:
    def __init__(self, x, y, subplot_value, has_error_bars):
        # This initialization will be called the LLM
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value
        self.has_error_bars = has_error_bars

    def estimate(self, image):
        pass

class Histogram:
    def __init__(self, x, y, subplot_value, has_error_bars):
        # This initialization will be called the LLM
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value
        self.has_error_bars = has_error_bars

    def estimate(self, image):
        pass

class Continuous:
    def __init__(self, x, y, subplot_value, has_error_bars):
        # This initialization will be called the LLM
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value
        self.has_error_bars = has_error_bars

    def estimate(self, image):
        pass