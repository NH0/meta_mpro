import numpy as np
from collections import defaultdict
from bisect import bisect
from time import time
from random import sample
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from itertools import combinations


from configuration import PATH, NAME

class Instance:
    radiuses = [(1,1),(1,2),(2,2),(2,3)]
    ks = [1,2,3]

    def __init__(self, name, ind_radius=0, ind_k=0, path=PATH):

        self._rcapt, self._rcom = Instance.radiuses[ind_radius][0], Instance.radiuses[ind_radius][1]
        self._k = Instance.ks[ind_k]
        self._data = {}
        self._data_x = None
        self._number_targets = 0
        self.read_instance(name, path)


    def read_instance(self, name, path=PATH):
        
        f = open(path + name,"r")
        for i, line in enumerate(f.readlines()):
            line = line.replace(";","")
            line = line.split()
            line = list(map(float,line[1:]))
            self._data[i] = line
            
        self._data_x = np.array(sorted(self._data.items(), key=lambda x: x[1][0]))
        # self._data_y = np.array(sorted(self._data.items(), key=lambda x: x[1][1]))
        self._n = len(self._data)
    
    def _distance_ind(self,i,j):
        
        return Instance.distance(self._data[i],self._data[j])
    
    @staticmethod
    def distance(x,y):
        
        return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    

class Solution(Instance):

    def __init__(self, name, ind_radius=0, ind_k=0, path=PATH):
        Instance.__init__(self, name, ind_radius, ind_k, path)
        self.sensors = nx.Graph()
        self.sensors.add_node(0)
        self.sensors_sorted = [[0,[0., 0.]]]
        self.counter = 0
        self.target_coverage = defaultdict(list)
        self.sensor_coverage = defaultdict(list)

    def _reduce_target(self,i):
        
        inf_x = bisect(self._data_x[:,1],list(np.array(self._data[i]) - np.array([self._rcapt,0])))
        sup_x = bisect(self._data_x[:,1],list(np.array(self._data[i]) + np.array([self._rcapt,0])))
        
        return inf_x, sup_x
        
    def _reduce_sensors(self,i):
        
        x = bisect(np.array(self.sensors_sorted)[:,1],self._data[i])
        inf_x = bisect(np.array(self.sensors_sorted)[:,1],list(np.array(self._data[i]) - np.array([self._rcom,0])))
        sup_x = bisect(np.array(self.sensors_sorted)[:,1],list(np.array(self._data[i]) + np.array([self._rcom,0])))
        
        return x, inf_x, sup_x
        
    def add_sensor(self,i):
        
        self.sensors.add_node(i)
        x, inf_x, sup_x = self._reduce_sensors(i)
        self.sensors_sorted.insert(x, [i,self._data[i]])
        
        for j in range(max(inf_x-1,0), min(sup_x+1,len(self.sensors_sorted))):
            x_j = self.sensors_sorted[j][0]
            if self._distance_ind(i,x_j) < self._rcom and i != x_j:
                self.sensors.add_edge(i,x_j)
                
        # for j in self.sensors.nodes:
        #     if self._distance_ind(i,j) < self._rcom:
        #         self.sensors.add_edge(i,j)
        
        inf_x, sup_x = self._reduce_target(i)
        self.counter += sup_x-inf_x
        for j in range(max(inf_x-1,0), min(sup_x+1,self._n)):
            x_j = self._data_x[j][0]
            if self._distance_ind(i,x_j) < self._rcapt:
                self.target_coverage[x_j].append(i)
                self.sensor_coverage[i].append(x_j)
    
                
    
    def remove_sensor(self,i):
        
        self.sensors.remove_node(i)
        self.sensors_sorted.remove([i,self._data[i]])
        del self.sensor_coverage[i]
        
        inf_x, sup_x = self._reduce_target(i)    
        for j in range(max(inf_x-1,0), min(sup_x+1,self._n)):
            x_j = self._data_x[j][0]
            if i in self.target_coverage[x_j]:
                self.target_coverage[x_j].remove(i)
                
    def to_be_removed(self,min_coverage=0,r=0):
        
        sensor_to_be_removed = []
        if min_coverage>0:
            for sensor in list(self.sensors.nodes)[1:]:
                L = list(map(lambda target:len(self.target_coverage[target]),self.sensor_coverage[sensor]))
                if min(L) == min_coverage-r:
                    sensor_to_be_removed.append(sensor)
        else:
            for sensor in list(self.sensors.nodes)[1:]:
                L = list(map(lambda target:len(self.target_coverage[target]),self.sensor_coverage[sensor]))
                if min(L) > min_coverage:
                    sensor_to_be_removed = [sensor]
                    min_coverage = min(L)
                elif min(L) == min_coverage:
                    sensor_to_be_removed.append(sensor)
                
        return min_coverage, sensor_to_be_removed
                
    def is_admissible(self):
        
        for i in range(1,self._n):
            if len(self.target_coverage[i]) < self._k:
                return False
                
        return nx.is_connected(self.sensors)
        
    def score(self):
        
        return len(self.sensors.nodes)-1
        
    def optimize_locally(self, r_max=2):
        
        admissible = True
        while admissible:
            admissible = self.is_removable_through_r(r_max)
    
    def is_removable_through_r(self, r_max):
        
        min_coverage=0
        for r in range(r_max):
            min_coverage, to_remove = self.to_be_removed(min_coverage,r)
            if self.is_removable(to_remove):
                return True
        return False
                
    def is_removable(self, to_remove):
        
        for i in to_remove:
            self.remove_sensor(i)
            if self.is_admissible():
                return True
            else:
                self.add_sensor(i)
        return False
        
    def voisinage(self):
        
        max_coverage = 0
        i_max = []
        for i in range(1,self._n):
            if len(self.target_coverage[i]) == max_coverage:
                i_max.append(i)
            if len(self.target_coverage[i]) > max_coverage:
                i_max = [i]
                max_coverage = len(self.target_coverage[i])    

        for i_test in i_max:
            to_test = self.target_coverage[i_test][:]
            if i_test in self.sensors.nodes:
                switch = list(combinations(self.sensor_coverage[i_test], len(self.target_coverage[i_test])-1))
            else:
                self.add_sensor(i_test)
                targets = self.sensor_coverage[i_test][:]
                self.remove_sensor(i_test)
                switch = list(combinations(targets, len(self.target_coverage[i_test])-1))
            if len(switch) > 5000:
                switch = sample(switch,5000)
            for sensor in to_test:
                self.remove_sensor(sensor)
        
            if not self.is_switchable(switch):
                print("fail")
                for j in to_test:
                    self.add_sensor(j)
            else:
                print("score: ",self.score())
                return 1
            
        return 0
    
    def is_switchable(self,switch):
        
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
                
        
            
    def optimize_voisi(self):
        
        voisi = True
        while voisi:
            voisi = Solution1.voisinage()
        
                
    def plot_sensors(self):

        _, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        
        for i in range(self._n):
            plt.scatter(*self._data[i],c="b")
        
        for i in self.sensors.nodes:
            x, y = self._data[i][0], self._data[i][1]
            plt.scatter(x, y, c="r")
            circle = ptc.Circle((x, y), radius=self._rcapt, ec="g", fc=(0,0,0,0.001), lw=0.5)
            ax.add_artist(circle)
        
        
        plt.show()

        
            ##

def quartering(solution, size=20):
    return 0
        

Solution1 = Solution(NAME,0,2)
t = time()
for i in range(1,1500):
    Solution1.add_sensor(i)
t1 = time()
print(t1-t)
print(Solution1.is_admissible())
t2 = time()
print(t2-t1)
Solution1.optimize_locally()
t3 = time()
print(Solution1.score())
print(t3-t2)
t4 = time()
Solution1.optimize_voisi()
print(t4-t3)
print(Solution1.score())
Solution1.plot_sensors()
Solution1.optimize_locally()

# 
# compteur = 0
# score = 0
# score_min = 1500
# for j in range(50):
#     Solution1 = Solution(NAME)
#     print(j)
#     for i in sample(range(1,1500),1400):
#         Solution1.add_sensor(i)
#     if Solution1.is_admissible():
#         compteur += 1
#         Solution1.optimize_locally()
#         score += Solution1.score()
#         if Solution1.score() < score_min:
#             score_min = Solution1.score()
#         print(Solution1.score())
# print(score/compteur)











