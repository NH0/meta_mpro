import vns
import Solution
import Instance
from time import time

from configuration import NAME
from quartering import quartering

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

Solution1 = Solution.Solution(NAME)
t = time()
Solution1.add_all()
t1 = time()
print("time to add everything : ", t1 - t)
t2 = time()
Solution1.optimize_locally()
print("time to optimize locally : ", t2 - t1)
print("first score : ", Solution1.score)
t3 = time()
Solution1.almost_annealing()
print("time to optimize with first meta : ", t3 - t2)
# quartering(Solution1)
# Solution1.optimize_voisi()
t4 = time()
print(t4 - t3)
print(Solution1.score)
Solution2, scores = vns.start_vns(Solution1)
t5 = time()
print("VNS time : ", t5 - t4)
print("Initial score ", Solution1.score)
print("VNS score ", Solution2.score)
print("Solution sensors ", list(Solution2.sensors.nodes))
Solution1.plot_sensors()
Solution2.plot_sensors()
# Solution1.optimize_locally()

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
#         score += Solution1.score
#         if Solution1.score < score_min:
#             score_min = Solution1.score
#         print(Solution1.score)
# print(score/compteur)
