import utils as utils

def quartering(solution, size=20):
    x_minimum = solution._data_x[0][1][0]
    x_maximum = solution._data_x[-1][1][0]
    y_minimum = solution._data_y[0][1][1]
    y_maximum = solution._data_y[-1][1][1]

    length = int(x_maximum - x_minimum) + 1
    width = int(y_maximum - y_minimum) + 1

    length_quarter = length * size / 100
    width_quarter = width * size / 100

    quarters = [[[] for j in range(width)] for i in range(length)]

    for target in solution._data.items():
        quarters[int((target[1][0] - x_minimum) / length_quarter)]\
            [int((target[1][1] - y_minimum) / width_quarter)].append(target)
