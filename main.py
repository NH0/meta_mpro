import vns
import Solution
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
Solution1.optimize_locally()
t2 = time()
print("time to optimize locally : ", t2 - t1)
print("first score : ", Solution1.score)
Solution1.almost_annealing()
t3 = time()
print("time to optimize with first meta : ", t3 - t2)
# quartering(Solution1)
# Solution1.optimize_voisi()
# t4 = time()
# print(t4-t3)
# print(Solution1.score)
# vns.start_vns(Solution1)
# t5 = time()
# print("VNS time : ", t5 - t4, end="")
# print(Solution1.score)
# Solution1.plot_sensors()

# nb = 5
# score = 0
# score_min = 1500
# for j in range(nb):
#     Solution1 = Solution.Solution(NAME)
#     print("#################################")
#     print("#################################")
#     print(j)
#     print("#################################")
#     print("#################################")
#     Solution1.add_all()
#     Solution1.optimize_locally()
#     Solution1.almost_annealing(100)
#     score += Solution1.score
#     if Solution1.score < score_min:
#         score_min = Solution1.score
#     print(Solution1.score)
# print(score/nb)
