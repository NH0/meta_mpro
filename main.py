import numpy as np


class capteurs:
    
    def __init__(self):
        
        self.L_Radius = np.array([(1,1),(1,2),(2,2),(2,3)])
        self.L_k = np.array([1,2,3])

    def read_instance(self,name, path = "/home/jeffrey/Bureau/cours/3a/meta-heuristique/"):
        
        f = open(path + name,"r")
        self.data = []
        for line in f.readlines():
            line = line.replace(";","")
            line = line.split()
            line = list(map(float,line))
            self.data.append(line)
            
        self.data = np.array(self.data)
            

Capteur = capteurs()
Capteur.read_instance("instance_1.dat")