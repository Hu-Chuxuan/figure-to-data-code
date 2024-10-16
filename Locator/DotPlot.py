class DotPlot(Subplot):
    def __init__(self, x, y, subplot_value, has_error_bars, value_direction):
        super().__init__(x, y, subplot_value, has_error_bars, value_direction)

    def estimate(self, image):
        raise NotImplementedError
