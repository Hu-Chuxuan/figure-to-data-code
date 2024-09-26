class Curve:
    def __init__(self, x, y, curve_value, error_bars=None):
        self.x = x
        self.y = y
        self.error_bars = error_bars
        self.curve_value = curve_value

class DotPlot:
    def __init__(self, x, y, subplot_value):
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value

    def estimate(self, has_error_bars):
        curves = []
        for subplot in self.subplots:
            curves.append(subplot.estimate_dot_points(has_error_bars))

class Histogram:
    def __init__(self, x, y, subplot_value):
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value

    def estimate(self, has_error_bars):
        curves = []
        for subplot in self.subplots:
            curves.append(subplot.estimate_histogram(has_error_bars))

class Continuous:
    def __init__(self, x, y, subplot_value):
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value

    def estimate(self, has_error_bars):
        curves = []
        for subplot in self.subplots:
            curves.append(subplot.estimate_continuous(has_error_bars))