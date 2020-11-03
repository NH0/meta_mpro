import concurrent.futures
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import copy
import time
from collections import defaultdict
from bisect import bisect
from random import sample, shuffle, random
from itertools import combinations

import logging
import inspect

import utils

from Instance import Instance
from configuration import PATH


class Solution(Instance):

    def __init__(self, name, ind_radius=0, ind_k=0, path=PATH):
        Instance.__init__(self, name, ind_radius, ind_k, path)
        self.sensors = nx.Graph()
        self.sensors.add_node(0)
        self.neighbors = nx.Graph()
        self.sensors_sorted = [[0, [0., 0.]]]
        self.target_coverage = defaultdict(list)

    @property
    def score(self):

        return len(self.sensors.nodes) - 1

    def copy(self):

        return copy.deepcopy(self)

    # Trouver les cibles proches d'une cible i
    def _reduce_target(self, i):

        inf_x = bisect(self._data_x[:, 1], list(
            np.array(self._data[i]) - np.array([self._rcapt, 0])))
        sup_x = bisect(self._data_x[:, 1], list(
            np.array(self._data[i]) + np.array([self._rcapt, 0])))

        return inf_x, sup_x

    def _find_neighbors(self, i, distance=1):
        neighbors = []
        inf_x, sup_x = self._reduce_target(i)
        for j in range(max(inf_x - 1, 1), min(sup_x + 1, self._n)):
            x_j = self._data_x[j][0]
            if self._distance_ind(i, x_j) <= distance:
                neighbors.append(x_j)

        return neighbors

    def _reduce_sensors(self, i):

        x = bisect(np.array(self.sensors_sorted, dtype=object)
                   [:, 1], self._data[i])
        inf_x = bisect(np.array(self.sensors_sorted, dtype=object)[:, 1], list(
            np.array(self._data[i]) - np.array([self._rcom, 0])))
        sup_x = bisect(np.array(self.sensors_sorted, dtype=object)[:, 1], list(
            np.array(self._data[i]) + np.array([self._rcom, 0])))

        return x, inf_x, sup_x

    # Ajouter ou supprimer des capteurs
    def add_sensor(self, i):

        if i in self.sensors.nodes:
            logging.warning(" {} already a sensor ...\n\t{}".format(
                i, inspect.getframeinfo(inspect.currentframe().f_back)))
            return 0

        self.sensors.add_node(i)
        x, inf_x, sup_x = self._reduce_sensors(i)
        self.sensors_sorted.insert(x, [i, self._data[i]])

        for j in range(max(inf_x - 1, 0), min(sup_x + 1, len(self.sensors_sorted))):
            x_j = self.sensors_sorted[j][0]
            if self._distance_ind(i, x_j) <= self._rcom and i != x_j:
                self.sensors.add_edge(i, x_j)

        neighbors = list(self.neighbors.neighbors(i))
        for x_j in neighbors:
            if i in self.target_coverage[x_j]:
                logging.warning(" {} already in target_coverage of {} ...\n\t{}\n{}\n{}".format(
                    i, x_j,
                    inspect.getframeinfo(inspect.currentframe().f_back),
                    inspect.getframeinfo(inspect.currentframe().f_back.f_back),
                    inspect.getframeinfo(inspect.currentframe().f_back.f_back.f_back)))
                continue
            self.target_coverage[x_j].append(i)

        return 1

    def remove_sensor(self, i):

        self.sensors.remove_node(i)
        self.sensors_sorted.remove([i, self._data[i]])

        neighbors = list(self.neighbors.neighbors(i))
        for x_j in neighbors:
            self.target_coverage[x_j] = list(filter(lambda x: x != i, self.target_coverage[x_j]))

        return 1

    # Check admissibility
    def _is_connected(self):

        return nx.is_connected(self.sensors)

    def _is_covered(self):
        for i in range(1, self._n):
            if len(self.target_coverage[i]) < self._k:

                return False, i

        return True, -1

    def is_admissible(self):

        return self._is_connected() and self._is_covered()[0]

    # Fonctions relatives à l'optimisation locale
    def add_all(self):

        for i in range(1, self._n):
            self.sensors.add_node(i)
            x, inf_x, sup_x = self._reduce_sensors(i)
            self.sensors_sorted.insert(x, [i, self._data[i]])

            for j in range(max(inf_x - 1, 0), min(sup_x + 1, len(self.sensors_sorted))):
                x_j = self.sensors_sorted[j][0]
                if self._distance_ind(i, x_j) <= self._rcom and i != x_j:
                    self.sensors.add_edge(i, x_j)

            neighbors = self._find_neighbors(i, distance=self._rcapt)
            for x_j in neighbors:
                self.target_coverage[x_j].append(i)
                self.neighbors.add_edge(i, x_j)

    def _to_be_removed(self, min_coverage=0, r=0):

        sensor_to_be_removed_1 = []
        if min_coverage > 0:
            for sensor in list(self.sensors.nodes)[1:]:
                L1 = list(map(lambda target: len(
                    self.target_coverage[target]), list(self.neighbors.neighbors(sensor))))

                if min(L1) == min_coverage - r:
                    sensor_to_be_removed_1.append(sensor)

        else:
            for sensor in list(self.sensors.nodes)[1:]:
                L1 = list(map(lambda target: len(
                    self.target_coverage[target]), list(self.neighbors.neighbors(sensor))))

                if min(L1) > min_coverage:
                    sensor_to_be_removed_1 = [sensor]
                    min_coverage = min(L1)
                elif min(L1) == min_coverage:
                    sensor_to_be_removed_1.append(sensor)

        shuffle(sensor_to_be_removed_1)

        return min_coverage, sensor_to_be_removed_1

    def _is_removable(self, to_remove):

        for i in to_remove:
            self.remove_sensor(i)
            if self.is_admissible():
                return True, i
            else:
                self.add_sensor(i)
        return False, 0

    def _is_removable_through_r(self, r_max):

        min_coverage = 0
        for r in range(r_max):
            min_coverage, to_remove = self._to_be_removed(min_coverage, r)
            admissible, removed = self._is_removable(to_remove)
            if admissible:
                return True, removed
        return False, 0

    # Optimisation locale
    def optimize_locally(self, r_max=2):

        admissible = True
        L_removed = []
        while admissible:
            admissible, removed = self._is_removable_through_r(r_max)
            if admissible:
                L_removed.append(removed)

        return L_removed

    # Fonctions relatives au voisinage V_2
    def _find_max_coverage(self, max_coverage, q):
        i_max = []
        if max_coverage == 0:
            for i in range(1, self._n):
                if len(self.target_coverage[i]) == max_coverage:
                    i_max.append(i)
                if len(self.target_coverage[i]) > max_coverage:
                    i_max = [i]
                    max_coverage = len(self.target_coverage[i])
        else:
            for i in range(1, self._n):
                if len(self.target_coverage[i]) == max_coverage - q:
                    i_max.append(i)
        return i_max, max_coverage

    def _is_switchable(self, switch):

        for i in range(len(switch)):
            try_switch = switch[i]
            for sensor in try_switch:
                if sensor == 0:
                    print("gros probleme")
                self.add_sensor(sensor)
            if not self.is_admissible():
                for sensor in try_switch:
                    if sensor == 0:
                        print("problem")
                    self.remove_sensor(sensor)
            else:
                return True
        return False

    def re_organize(self, nb_reorganized):

        to_change = sample(range(1, self._n), nb_reorganized)

        for i_test in to_change:
            to_test = self.target_coverage[i_test][:]
            logging.debug(
                "Removing sensors {}\tTarget node is {}".format(to_test, i_test))

            switch = list(combinations(self.neighbors.neighbors(
                i_test), len(self.target_coverage[i_test])))
            shuffle(switch)

            for sensor in to_test:
                self.remove_sensor(sensor)
            if not self._is_switchable(switch):
                logging.debug(
                    "Neighborhood 1 : Switch fail around {}".format(i_test))
                for sensor in to_test:
                    self.add_sensor(sensor)

    # Voisinage V_1
    def neighborhood_v_1(self, nb_added):

        score = self.score
        addable = [i for i in range(1, self._n) if i not in self.sensors.nodes]
        to_add = sample(addable, nb_added)
        for sensor in to_add:
            self.add_sensor(sensor)
        removed = self.optimize_locally()
        if self.score < score:
            score = self.score
            logging.debug("Neighborhood 3 : {}", score)
            return True
        else:
            for sensor in removed:
                if sensor == 0:
                    print("removed")
                self.add_sensor(sensor)
            for sensor in to_add:
                self.remove_sensor(sensor)
            return False

    # Fonctions relatives au voisinage V_2

    def add_sensor_close_to_target(self, target_index):
        if target_index in self.sensors.nodes or target_index == 0:
            index_neighbors = np.array(
                sorted(
                    self._data.items(),
                    key=lambda x: utils.distance(
                        x[1],
                        self._data[target_index])),
                dtype=object)
            i = 0
            while i < len(index_neighbors) and (index_neighbors[i][0] in self.sensors.nodes):
                i += 1
            if i == len(index_neighbors):
                logging.warning("Cannot add sensor close to {}", target_index)
                return -1
            self.add_sensor(index_neighbors[i][0])
            logging.debug("close to target : added sensor (neighbor of {}) {}".format(
                target_index, index_neighbors[i][0]))
        else:
            self.add_sensor(target_index)
            logging.debug(
                "close to target : added sensor {}".format(target_index))

    def remove_random_sensors(self, nb_removed):
        random_generator = np.random.default_rng()

        if 0 in self.sensors.nodes:
            list_choices = list(self.sensors.nodes)
            list_choices.remove(0)
            to_be_removed = random_generator.choice(list_choices,
                                                    size=nb_removed,
                                                    replace=False,
                                                    shuffle=True)
        else:
            to_be_removed = random_generator.choice(list(self.sensors.nodes),
                                                    size=nb_removed,
                                                    replace=False,
                                                    shuffle=True)
        logging.debug(
            "Neighborhood 2 : Removing {} sensors".format(to_be_removed))

        for sensor in to_be_removed:
            self.remove_sensor(sensor)

    def fix_broken_coverrage(self):
        nb_added = 0
        is_covered, index_not_covered = self._is_covered()
        while not(is_covered):
            if index_not_covered == 0:
                raise ValueError(index_not_covered)
            self.add_sensor_close_to_target(target_index=index_not_covered)
            nb_added += 1
            is_covered, index_not_covered = self._is_covered()

        return nb_added

    def fix_broken_connection(self):
        nb_added = 0
        random_generator = np.random.default_rng()
        while not(self._is_connected()):
            connected_components = [list(self.sensors.subgraph(
                component).nodes) for component in nx.connected_components(self.sensors)]
            logging.debug("Neighborhood 2 : {} connected components".format(
                len(list(connected_components))))
            neighbors00 = self._find_neighbors(0, distance=self._rcom)
            found_component = False
            for neighbor in neighbors00:
                if neighbor in self.sensors.nodes:
                    component_with_00 = list(nx.node_connected_component(
                        self.sensors, neighbor))
                    found_component = True
                    break
            if not found_component:
                self.add_sensor(neighbors00[0])
                component_with_00 = list(nx.node_connected_component(
                    self.sensors, neighbors00[0]))

            # Choose the closest component X to the component Y containing (0,0)
            random_element_connected_to_00 = random_generator.choice(
                component_with_00)
            smallest_distance_to_random_element = np.inf
            closest_component = None
            for component in connected_components:
                if random_element_connected_to_00 in component:
                    continue
                for node in component:
                    distance_to_random_element = self._distance_ind(
                        node, random_element_connected_to_00)
                    if distance_to_random_element < smallest_distance_to_random_element:
                        smallest_distance_to_random_element = distance_to_random_element
                        closest_component = component
            closest_component = list(closest_component)

            # Choose the closest node y0 to a random x of X, in the component (0,0)
            random_element_component = random_generator.choice(
                closest_component)
            smallest_distance_to_component = np.inf
            closest_node = 0
            for node in component_with_00:
                distance_to_component = self._distance_ind(
                    node, random_element_component)
                if distance_to_component < smallest_distance_to_component:
                    smallest_distance_to_component = distance_to_component
                    closest_node = node

            # Choose the closest node to y0, in the component X
            smallest_distance_to_closest_node = smallest_distance_to_component
            closest_node_in_component = random_element_component
            for node in component:
                distance_to_closest_node = self._distance_ind(
                    node, closest_node)
                if distance_to_closest_node < smallest_distance_to_closest_node:
                    smallest_distance_to_closest_node = distance_to_closest_node
                    closest_node_in_component = node

            barycenter = utils.compute_barycenter(
                [[closest_node,
                  self._data[closest_node]],
                 [closest_node_in_component,
                  self._data[closest_node_in_component]]])
            closest_target = utils.find_closest(
                barycenter, list(self._data.items()))
            self.add_sensor_close_to_target(closest_target[0])
            nb_added += 1

        return nb_added

    # Voisinage V_2
    def neighborhood_2(self, to_remove=4):
        """
        Remove to_remove sensors randomly selected.
        Then fix the coverage and connection for admissibility
        """
        self.remove_random_sensors(to_remove)

        nb_added = 0
        nb_added += self.fix_broken_connection()
        nb_added += self.fix_broken_coverrage()
        nb_added += self.fix_broken_connection()

        if not(self.is_admissible()):
            raise RuntimeError

        if to_remove - nb_added > 0:
            logging.debug("Neighborhood 2 : Removed {} sensors".format(to_remove - nb_added))
        else:
            logging.debug("Neighborhood 2 : Added {} sensors".format(nb_added - to_remove))

        return to_remove - nb_added

    # Display solution
    def plot_sensors(self):

        _, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        for i in range(self._n):
            plt.scatter(*self._data[i], c="b")

        for i in self.sensors.nodes:
            x, y = self._data[i][0], self._data[i][1]
            plt.scatter(x, y, c="r")
            circle = ptc.Circle((x, y), radius=self._rcapt,
                                ec="g", fc=(0, 0, 0, 0.001), lw=0.5)
            ax.add_artist(circle)

        plt.show()
