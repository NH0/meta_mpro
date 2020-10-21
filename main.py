import numpy as np
from collections import defaultdict
from bisect import bisect
from time import time
from random import sample

path = "/home/jeffrey/Bureau/cours/3a/meta-heuristique/Instances/"
name = "captANOR1500_18_100.dat"


class Project:
    
    def __init__(self):
        
        self.L_Radius = [(1,1),(1,2),(2,2),(2,3)]
        self.L_k = [1,2,3]
        self.L_capteurs = []
        self.compteur = 0
        
    def chose(self, ind_radius, ind_k):
        
        self._Rcapt, self._Rcom = self.L_Radius[ind_radius][0], self.L_Radius[ind_radius][1]
        self._k = self.L_k[ind_k]

    def read_instance(self,name):
        
        f = open(path + name,"r")
        self._data = {}
        self.capteurs_to_targets = []
        for i, line in enumerate(f.readlines()):
            
            line = line.replace(";","")
            line = line.split()
            
            line = list(map(float,line[1:]))
            self._data[i] = line
            
        self._data_x = np.array(sorted(self._data.items(), key=lambda x: x[1][0]))
        # self._data_y = np.array(sorted(self._data.items(), key=lambda x: x[1][1]))
        self._n = len(self._data)
        self.capteurs_to_targets = defaultdict(list)
    
    def _distance_ind(self,i,j):
        
        return self._distance(self._data[i],self._data[j])
        
    def _distance(self,x,y):
        
        return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    
    def _reduce(self,i):
        
        inf_x = bisect(self._data_x[:,1],list(np.array(self._data[i]) - np.array([self._Rcapt,0])))
        sup_x = bisect(self._data_x[:,1],list(np.array(self._data[i]) + np.array([self._Rcapt,0])))
        return inf_x, sup_x
        
    def add_capteur(self,i):
        
        self.L_capteurs.append(i)
        inf_x, sup_x = self._reduce(i)
        self.compteur += sup_x-inf_x
        for j in range(inf_x, sup_x):
            x_j = self._data_x[j][0]
            if self._distance_ind(i,x_j) < self._Rcapt:
                self.capteurs_to_targets[x_j].append(i)
        # for j in range(self._n):
        #     if self._distance(i,j) < self._Rcapt:
        #         self.capteurs_to_targets[j].append(i)
    
    def remove_capteur(self,i):
        
        if i in self.L_capteurs:
            self.L_capteurs.remove(i)
        else:
            print("erreur")
        inf_x, sup_x = self._reduce(i)    
        for j in range(inf_x, sup_x):
            x_j = self._data_x[j][0]
            if i in self.capteurs_to_targets[x_j]:
                self.capteurs_to_targets[x_j].remove(i)
            
    def is_admissible(self):
        
        i = 1
        while i < self._n and len(self.capteurs_to_targets[i]) >= self._k:
            i += 1
        print(i)   
        bool1 = (i == self._n) 
        
        return bool1
            
        

Project1 = Project()
Project1.chose(0,0)
Project1.read_instance(name)
t = time()
for i in sample(range(1,1500),1000):
    Project1.add_capteur(i)
t1 = time()
print(t1-t)
print(Project1.is_admissible())
t2 = time()
print(t2-t1)

























