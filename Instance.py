import numpy as np

import utils as utils

from configuration import PATH


class Instance:
    radiuses = [(1, 1), (1, 2), (2, 2), (2, 3)]
    ks = [1, 2, 3]

    def __init__(self, name, ind_radius=0, ind_k=0, path=PATH, readinstance=True):

        self._rcapt = Instance.radiuses[ind_radius][0]
        self._rcom = Instance.radiuses[ind_radius][1]
        self._k = Instance.ks[ind_k]
        self.name = name
        self._data = {}
        self._data_x = None
        self._number_targets = 0
        if readinstance:
            self.read_instance(name, path)

    def read_instance(self, name, path=PATH):

        with open(path + name, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.replace(";", "")
                line = line.split()
                line = list(map(float, line[1:]))
                self._data[i] = line

            self._data_x = np.array(
                sorted(self._data.items(), key=lambda x: x[1][0]), dtype=object)
            self._data_y = np.array(
                sorted(self._data.items(), key=lambda x: x[1][1]), dtype=object)
            self._n = len(self._data)

    @staticmethod
    def generate_instance_from_existing(name, path=PATH, size=3.0):
        instance = Instance(name, path=path, readinstance=False)

        with open(path + name, "r") as f:
            i = 0
            for line in f:
                line = line.replace(";", "")
                line = line.split()
                line = list(map(float, line[1:]))
                if line[0] < size and line[1] < size:
                    instance._data[i] = line
                    i += 1

            instance._data_x = np.array(
                sorted(instance._data.items(), key=lambda x: x[1][0]), dtype=object)
            instance._data_y = np.array(
                sorted(instance._data.items(), key=lambda x: x[1][1]), dtype=object)
            instance._n = len(instance._data)

        return instance

    def save_instance_to_file(self, name, path=PATH):

        with open(path + name, "x") as f:
            pass
        with open(path + name, "w") as f:
            for target in self._data.items():
                f.write("{}  {} {}\n".format(target[0], target[1][0], target[1][1]))

    def _distance_ind(self, i, j):

        return utils.distance(self._data[i], self._data[j])
