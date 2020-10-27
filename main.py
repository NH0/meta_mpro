import numpy as np
from collections import defaultdict
from bisect import bisect
from time import time
from random import sample

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
        self.sensors = []
        self.counter = 0
        self.target_coverage = defaultdict(list)

    def _reduce(self,i):
        
        inf_x = bisect(self._data_x[:,1],list(np.array(self._data[i]) - np.array([self._rcapt,0])))
        sup_x = bisect(self._data_x[:,1],list(np.array(self._data[i]) + np.array([self._rcapt,0])))
        return inf_x, sup_x
        
    def add_capteur(self,i):
        
        self.sensors.append(i)
        inf_x, sup_x = self._reduce(i)
        self.counter += sup_x-inf_x
        for j in range(inf_x, sup_x):
            x_j = self._data_x[j][0]
            if self._distance_ind(i,x_j) < self._rcapt:
                self.target_coverage[x_j].append(i)
        # for j in range(self._n):
        #     if self._distance(i,j) < self._Rcapt:
        #         self.target_coverage[j].append(i)
    
    def remove_capteur(self,i):
        
        self.sensors.remove(i)
        inf_x, sup_x = self._reduce(i)    
        for j in range(inf_x, sup_x):
            x_j = self._data_x[j][0]
            if i in self.target_coverage[x_j]:
                self.target_coverage[x_j].remove(i)
            
    def is_admissible(self):
        
        i = 1
        while i < self._n and len(self.target_coverage[i]) >= self._k:
            i += 1
        print(i)   
        bool1 = (i == self._n) 
        
        return bool1
            
        

Solution1 = Solution(NAME)
t = time()
for i in sample(range(1,150),100):
    Solution1.add_capteur(i)
t1 = time()
print(t1-t)
print(Solution1.is_admissible())
t2 = time()
print(t2-t1)

























