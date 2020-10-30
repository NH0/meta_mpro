import utils as utils

def compute_mean(list_targets, axis=0):
    axis_sum = 0
    for element in list_targets:
        axis_sum += element[1][axis]

    return axis_sum / len(element)

def compute_barycenter(list_targets):
    x_mean = compute_mean(list_targets, 0)
    y_mean = compute_mean(list_targets, 1)

    return [x_mean, y_mean]

def find_closest_target(list_targets):
    barycenter = compute_barycenter(list_targets)
    
    smallest_distance_to_barycenter = utils.distance(list_targets[0][1], barycenter)
    closest_target = list_targets[0]
    for target in list_targets[1:]:
        distance_to_barycenter = utils.distance(target[1], barycenter)
        if distance_to_barycenter < smallest_distance_to_barycenter:
            smallest_distance_to_barycenter = distance_to_barycenter
            closest_target = target
    
    return closest_target

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