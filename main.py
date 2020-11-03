import vns
import Solution
import Instance
from time import time

from configuration import NAMES, NAME
from quartering import quartering

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

RS = [(1, 1), (1, 2), (2, 2), (2, 3)]


def solve_with_almost_annealing():
    Solution1 = Solution.Solution(NAME)
    t = time()
    Solution1.add_all()
    t1 = time()
    print("time to add everything : ", t1 - t)
    Solution1.optimize_locally()
    t2 = time()
    print("time to optimize locally : ", t2 - t1)
    print("first score : ", Solution1.score)
    Solution1.almost_annealing(multiproc=False)
    # t3 = time()
    # print("time to optimize with first meta : ", t3 - t2)
    # Solution1.optimize_voisi()
    # t4 = time()
    # print(t4 - t3)
    # print(Solution1.score)
    # Solution1.plot_sensors()


def mean_almost_annealing():
    nb = 5
    score = 0
    score_min = 1500
    for j in range(nb):
        Solution1 = Solution.Solution(NAME)
        print("#################################")
        print("#################################")
        print(j)
        print("#################################")
        print("#################################")
        Solution1.add_all()
        Solution1.optimize_locally()
        Solution1.almost_annealing(100)
        score += Solution1.score
        if Solution1.score < score_min:
            score_min = Solution1.score
        print(Solution1.score)
    print(score / nb)


def solve_with_vns():
    with open("logs.txt", "a") as f:
        f.write("-" * 70 + "\n")
    for name in NAMES[5:]:
        for i in range(3):
            for j in range(4):
                print("---------------\n"
                      "Starting {name} with k={k} "
                      "(R_capt, R_com)=({r_capt},{r_com})".format(
                          name=name,
                          k=i + 1,
                          r_capt=RS[j][0],
                          r_com=RS[j][1]
                      ))
                Solution1 = Solution.Solution(name, ind_radius=j, ind_k=i)
                t = time()
                Solution1.add_all()
                t1 = time()
                print("time to add everything : ", t1 - t)
                Solution1.optimize_locally()
                t2 = time()
                print("time to optimize locally : ", t2 - t1)
                print("first score : ", Solution1.score)
                t3 = time()
                # Solution1.almost_annealing()
                print("time to optimize with first meta : ", t3 - t2)
                # Solution1.optimize_voisi()
                t4 = time()
                print(t4 - t3)
                print(Solution1.score)
                Solution2, scores = vns.start_vns(Solution1)
                t5 = time()
                print("VNS time : {:.1f}".format(t5 - t4))
                print("Initial score {}\t VNS score {}".format(Solution1.score, Solution2.score))
                print("Solution sensors ", list(Solution2.sensors.nodes))
                with open("logs.txt", "a") as f:
                    f.write("{instance} k={k} (R_capt, R_com)=({r_capt},{r_com}) "
                            "value={best_value} "
                            "time={time_exec:.1f}s : "
                            "{sensors}\n".format(instance=Solution2.name,
                                                 k=Solution2._k,
                                                 r_capt=Solution2._rcapt,
                                                 r_com=Solution2._rcom,
                                                 best_value=Solution2.score,
                                                 time_exec=t5 - t1,
                                                 sensors=sorted(list(Solution2.sensors.nodes))))
                # Solution1.plot_sensors()
                # Solution2.plot_sensors()


if __name__ == "__main__":
    # solve_with_vns()
    solve_with_almost_annealing()
    # mean_almost_annealing()
