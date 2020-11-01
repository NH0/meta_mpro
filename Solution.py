import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from collections import defaultdict
from bisect import bisect
from random import sample, shuffle
from itertools import combinations

import logging
import inspect

import utils

from Instance import Instance
from configuration import PATH


class Solution(Instance):

    def __init__(self, name, ind_radius=0, ind_k=0, path=PATH):
        Instance.__init__(self, name, ind_radius, ind_k, path)
        self.name = name
        self.sensors = nx.Graph()
        self.sensors.add_node(0)
        self.sensors_sorted = [[0, [0., 0.]]]
        self.counter = 0
        self.target_coverage = defaultdict(list)
        self.sensor_coverage = defaultdict(list)

    @property
    def score(self):

        return len(self.sensors.nodes)-1

    def copy(self):

        Solution_cop = Solution(self.name)
        Solution_cop.sensors = self.sensors.copy()
        Solution_cop.sensors_sorted = self.sensors_sorted[:]
        Solution_cop.target_coverage = self.target_coverage.copy()
        Solution_cop.sensor_coverage = self.sensor_coverage.copy()

        return Solution_cop

    # Add of remove sensors
    def _reduce_target(self, i):

        inf_x = bisect(self._data_x[:, 1], list(np.array(self._data[i])
                                                - np.array([self._rcapt, 0])))
        sup_x = bisect(self._data_x[:, 1], list(np.array(self._data[i])
                                                + np.array([self._rcapt, 0])))

        return inf_x, sup_x

    def _reduce_sensors(self, i):

        x = bisect(np.array(self.sensors_sorted, dtype=object)
                   [:, 1], self._data[i])
        inf_x = bisect(np.array(self.sensors_sorted, dtype=object)[:, 1], list(
            np.array(self._data[i]) - np.array([self._rcom, 0])))
        sup_x = bisect(np.array(self.sensors_sorted, dtype=object)[:, 1], list(
            np.array(self._data[i]) + np.array([self._rcom, 0])))

        return x, inf_x, sup_x

    def add_sensor(self, i):
        if i in self.sensors.nodes:
            logging.warning(" {} already a sensor ...\n\t{}".format(
                i, inspect.getframeinfo(inspect.currentframe().f_back)))
            return 0

        self.sensors.add_node(i)
        x, inf_x, sup_x = self._reduce_sensors(i)
        self.sensors_sorted.insert(x, [i, self._data[i]])

        for j in range(max(inf_x-1, 0), min(sup_x+1, len(self.sensors_sorted))):
            x_j = self.sensors_sorted[j][0]
            if self._distance_ind(i, x_j) <= self._rcom and i != x_j:
                self.sensors.add_edge(i, x_j)

        # for j in self.sensors.nodes:
        #     if self._distance_ind(i,j) < self._rcom:
        #         self.sensors.add_edge(i,j)

        inf_x, sup_x = self._reduce_target(i)
        self.counter += sup_x-inf_x
        for j in range(max(inf_x-1, 0), min(sup_x+1, self._n)):
            x_j = self._data_x[j][0]
            if self._distance_ind(i, x_j) <= self._rcapt:
                self.target_coverage[x_j].append(i)
                self.sensor_coverage[i].append(x_j)

        return 1

    def remove_sensor(self, i):

        self.sensors.remove_node(i)
        self.sensors_sorted.remove([i, self._data[i]])
        del self.sensor_coverage[i]

        inf_x, sup_x = self._reduce_target(i)
        for j in range(max(inf_x-1, 0), min(sup_x+1, self._n)):
            x_j = self._data_x[j][0]
            if i in self.target_coverage[x_j]:
                self.target_coverage[x_j].remove(i)

    # Check admissibility
    def is_connected(self):

        return nx.is_connected(self.sensors)

    def is_covered(self):
        for i in range(1, self._n):
            if len(self.target_coverage[i]) < self._k:

                return False, i

        return True, -1

    def is_admissible(self):

        return self.is_connected() and self.is_covered()[0]

    # Optimize locally the solution
    def to_be_removed(self, min_coverage=[0, 0], r=0):

        # sensor_to_be_removed_1, sensor_to_be_removed_2 = [], []
        sensor_to_be_removed_1 = []
        degrees = self.sensors.degree()
        if min_coverage > 0:
            for sensor in list(self.sensors.nodes)[1:]:
                L1 = list(map(lambda target: len(
                    self.target_coverage[target]), self.sensor_coverage[sensor]))
                # L2 = list(map(lambda target:degrees[target],self.target_coverage[sensor]))
                if min(L1) == min_coverage-r:
                    sensor_to_be_removed_1.append(sensor)
                # if min(L2) == min_coverage-r:
                #     sensor_to_be_removed_2.append(sensor)
        else:
            for sensor in list(self.sensors.nodes)[1:]:
                L1 = list(map(lambda target: len(
                    self.target_coverage[target]), self.sensor_coverage[sensor]))
                # L2 = list(map(lambda target:degrees[target],self.target_coverage[sensor]))
                if min(L1) > min_coverage:
                    sensor_to_be_removed_1 = [sensor]
                    min_coverage = min(L1)
                elif min(L1) == min_coverage:
                    sensor_to_be_removed_1.append(sensor)
                # if min(L2) > min_coverage:
                #     sensor_to_be_removed_2 = [sensor]
                #     min_coverage = min(L2)
                # elif min(L2) == min_coverage:
                #     sensor_to_be_removed_2.append(sensor)

        # sensor_to_be_removed = sensor_to_be_removed_1 + sensor_to_be_removed_2
        # shuffle(sensor_to_be_removed)
        shuffle(sensor_to_be_removed_1)

        return min_coverage, sensor_to_be_removed_1

    def is_removable(self, to_remove):

        for i in to_remove:
            self.remove_sensor(i)
            if self.is_admissible():
                return True, i
            else:
                self.add_sensor(i)
        return False, 0

    def is_removable_through_r(self, r_max):

        min_coverage = 0
        # min_coverage = [0,0]
        for r in range(r_max):
            min_coverage, to_remove = self.to_be_removed(min_coverage, r)
            admissible, removed = self.is_removable(to_remove)
            if admissible:
                return True, removed
        return False, 0

    def optimize_locally(self, r_max=2):

        admissible = True
        L_removed = []
        while admissible:
            admissible, removed = self.is_removable_through_r(r_max)
            if admissible:
                L_removed.append(removed)

        return L_removed

    # First neighborhood structure
    def find_max_coverage(self, max_coverage, q):
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

    def is_switchable(self, switch):

        for i in range(len(switch)):
            try_switch = switch[i]
            for sensor in try_switch:
                self.add_sensor(sensor)
            if not self.is_admissible():
                for sensor in try_switch:
                    self.remove_sensor(sensor)
            else:
                return True
        return False

    def voisinage1(self, max_coverage=0, q=0, nb_removed=1):
        """
        q : le q ième plus grand nombre de capteurs
        nb_removed : on retire p et on ajoute p-i
        """

        i_max, max_coverage = self.find_max_coverage(max_coverage, q)
        for i_test in i_max:
            to_test = self.target_coverage[i_test][:]
            logging.debug(
                "Removing sensors {}\tTarget node is {}".format(to_test, i_test))
            if i_test in self.sensors.nodes:
                switch = list(combinations(self.sensor_coverage[i_test], len(
                    self.target_coverage[i_test])-nb_removed))
            else:
                self.add_sensor(i_test)
                targets = self.sensor_coverage[i_test][:]
                self.remove_sensor(i_test)
                switch = list(combinations(targets, len(
                    self.target_coverage[i_test])-nb_removed))
            # if len(switch) > 5000:
            #     switch = sample(switch,5000)
            for sensor in to_test:
                self.remove_sensor(sensor)

            if not self.is_switchable(switch):
                logging.debug(
                    "Neighborhood 1 : Switch fail around {}".format(i_test))
                for sensor in to_test:
                    self.add_sensor(sensor)
            else:
                logging.debug("Neighborhood 1 : Switch sucess around {}\tNew score {}".format(
                    i_test, self.score()))
                return 1, max_coverage

        return 0, max_coverage

    # Second neighborhood structure
    def add_sensor_close_to_target(self, target_index):
        if target_index in self.sensors.nodes or target_index == 0:
            index_neighboors = np.array(sorted(self._data.items(),
                                               key=lambda x: utils.distance(x[1],
                                                                            self._data[target_index])), dtype=object)
            i = 0
            while index_neighboors[i][0] in self.sensors.nodes:
                i += 1
            self.add_sensor(index_neighboors[i][0])
            logging.debug("close to target : added sensor (neighbor of {}) {}".format(
                target_index, index_neighboors[i][0]))
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
                                                    size=nb_removed, replace=False, shuffle=True)
        else:
            to_be_removed = random_generator.choice(list(self.sensors.nodes),
                                                    size=nb_removed, replace=False, shuffle=True)
        logging.debug(
            "Neighborhood 2 : Removing {} sensors".format(to_be_removed))

        for sensor in to_be_removed:
            self.remove_sensor(sensor)

    def fix_broken_coverrage(self):
        nb_added = 0
        is_covered, index_not_covered = self.is_covered()
        while not(is_covered):
            self.add_sensor_close_to_target(target_index=index_not_covered)
            nb_added += 1
            is_covered, index_not_covered = self.is_covered()

        return nb_added

    def fix_broken_connection(self):
        nb_added = 0
        random_generator = np.random.default_rng()
        while not(self.is_connected()):
            connected_components = [list(self.sensors.subgraph(
                component).nodes) for component in nx.connected_components(self.sensors)]
            logging.debug("Neighborhood 2 : {} connected components".format(
                len(list(connected_components))))
            while len(self.target_coverage[0]) == 0:
                self.add_sensor_close_to_target(0)
            component_with_00 = list(nx.node_connected_component(
                self.sensors, self.target_coverage[0][0]))

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

            barycenter = utils.compute_barycenter([[closest_node, self._data[closest_node]],
                                                   [closest_node_in_component, self._data[closest_node_in_component]]])
            closest_target = utils.find_closest(
                barycenter, list(self._data.items()))
            self.add_sensor_close_to_target(closest_target[0])
            nb_added += 1

        return nb_added

    def voisinage2(self, nb_removed=4):
        self.remove_random_sensors(nb_removed)

        nb_added = 0
        nb_added += self.fix_broken_coverrage()
        nb_added += self.fix_broken_connection()

        if not(self.is_admissible()):
            raise RuntimeError

        logging.debug("Removed {} sensors".format(nb_removed - nb_added))

        return nb_removed - nb_added

    # Another local optimization using neighborhood1 structure
    def optimize_voisi(self):

        voisi = True
        while voisi:
            voisi, _ = self.voisinage1()

    # Third neighborhood structure
    def Neighborhood_3(self, T=20):

        score = self.score()
        addable = [i for i in range(self._n) if i not in self.sensors.nodes]
        to_add = sample(addable, T)
        for sensor in to_add:
            self.add_sensor(sensor)
        removed = self.optimize_locally()
        if self.score() < score:
            score = self.score()
            print("score : ", score)
            return True
        else:
            for sensor in removed:
                self.add_sensor(sensor)
            for sensor in to_add:
                self.remove_sensor(sensor)
            return False

    def almost_annealing(self):

        T = 20
        cmax = 500
        c = 0
        # Solution_save = self.copy()
        while T > 0 and c < cmax:

            self.Neighborhood_3()
            c += 1

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
