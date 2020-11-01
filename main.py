from time import time

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import vns
import Solution
from configuration import PATH, NAME
from quartering import quartering


Solution1 = Solution.Solution(NAME)
t = time()
number_of_targets = int(NAME[8:].partition('_')[0])
for i in range(1,number_of_targets):
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
# Solution1.optimize_voisi()
t4 = time()
print(t4-t3)
print(Solution1.score())
Solution2, scores = vns.start_vns(Solution1)
t5 = time()
print("VNS time : ",t5-t4, end="")
print(Solution1.score())
print("Score : ", Solution2.score())
Solution1.plot_sensors()
Solution2.plot_sensors()
#Solution1.optimize_locally()

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
