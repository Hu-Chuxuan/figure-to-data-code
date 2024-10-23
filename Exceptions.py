YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

class WrongCSVNumberError(Exception):
    def __init__(self, pred_num, gt_num):
        super().__init__(YELLOW + f"The number of predicted CSV files is {pred_num} when the ground truth has {gt_num} CSV files." + RESET)

class FormatError(Exception):
    def __init__(self, msg):
        super().__init__(YELLOW + msg + RESET)
