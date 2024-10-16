class Curve:
    def __init__(self, color, shape, x, y, curve_value, error_bars=None):
        # Must contain the color and shape of the curve to distinguish it from other curves
        self.color = color
        self.shape = shape
        self.x = x
        self.y = y
        self.error_bars = error_bars
        self.curve_value = curve_value

class Subplot:
    def __init__(self, x, y, subplot_value, has_error_bars, value_direction):
        # This initialization will be called the LLM
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value
        self.has_error_bars = has_error_bars
        self.value_direction = value_direction
        self.curves = []
    
    def estimate(self, image):
        raise NotImplementedError

    def to_value(self):
        raise NotImplementedError



