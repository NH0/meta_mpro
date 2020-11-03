import numpy as np

# Compute the distance between x1 and x2


def distance(x1, x2):

    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

# find the closest element to x in a list of elements


def find_closest(x, list_of_xs):
    smallest_distance = distance(x[1], list_of_xs[0][1])
    closest = list_of_xs[0]

    for x2 in list_of_xs[1:]:
        distance_between_x_x2 = distance(x[1], x2[1])
        if distance_between_x_x2 < smallest_distance:
            smallest_distance = distance_between_x_x2
            closest = x2

    return closest

# compute the mean of elements over and axis


def compute_mean(list_targets, axis=0):
    axis_sum = 0
    for element in list_targets:
        axis_sum += element[1][axis]

    return axis_sum / len(element)

# compute the barycenter of a list of targets


def compute_barycenter(list_targets):
    x_mean = compute_mean(list_targets, 0)
    y_mean = compute_mean(list_targets, 1)

    return [-1, [x_mean, y_mean]]

# find the closest element in a list of target to the barycenter of the list


def find_closest_target_to_barycenter(list_targets):
    barycenter = compute_barycenter(list_targets)

    return find_closest(barycenter, list_targets)
