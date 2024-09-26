class Curve:
    def __init__(self, x, y, curve_value, error_bars=None):
        self.x = x
        self.y = y
        self.error_bars = error_bars
        self.curve_value = curve_value

class Plot:
    def __init__(self, subplots):
        self.subplots = subplots
    
    def estimate_dot_points(self, has_error_bars):
        curves = []
        for subplot in self.subplots:
            curves.append(subplot.estimate_dot_points(has_error_bars))

    def estimate_histogram(self, has_error_bars):
        curves = []
        for subplot in self.subplots:
            curves.append(subplot.estimate_histogram(has_error_bars))

    def estimate_continuous(self, has_error_bars):
        curves = []
        for subplot in self.subplots:
            curves.append(subplot.estimate_continuous(has_error_bars))

class Subplot:
    def __init__(self, x, y, subplot_value):
        self.x_axis = x
        self.y_axis = y
        self.subplot_value = subplot_value

    def estimate_dot_points(self):
        pass

    def estimate_histogram(self):
        pass

    def estimate_continuous(self):
        pass