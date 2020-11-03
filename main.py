import logging
import vns
import Solution
from time import time

PATH = "Instances/"
NAMES = ["captANOR150_7_4.dat",
         "captANOR225_8_10.dat",
         "captANOR625_12_100.dat",
         "captANOR900_15_20.dat",
         "captANOR1500_15_100.dat",
         "captANOR1500_18_100.dat"]
NAME = NAMES[-1]

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

RS = [(1, 1), (1, 2), (2, 2), (2, 3)]


K = 1

Indice_R_capt_R_com = 0
R_capt, R_com = RS[Indice_R_capt_R_com]


def solve_with_vns(plot_solution=True):
    print("---------------\n"
          "Starting {name} with k={k} "
          "(R_capt, R_com)=({r_capt},{r_com})".format(
              name=NAME,
              k=K,
              r_capt=R_capt,
              r_com=R_com
          ))
    Solution1 = Solution.Solution(NAME, ind_radius=Indice_R_capt_R_com, ind_k=K)
    t = time()
    Solution1.add_all()
    t1 = time()
    print("time to add everything : {:.2f}s".format(t1 - t))
    Solution1.optimize_locally()
    t2 = time()
    print("time to optimize locally : {:.2f}s".format(t2 - t1))
    print("first score : ", Solution1.score)
    t3 = time()
    Solution2, scores = vns.start_vns(Solution1)
    t4 = time()
    print("VNS time : {:.1f}s".format(t4 - t3))
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
                                     time_exec=t4 - t1,
                                     sensors=sorted(list(Solution2.sensors.nodes))))
    if plot_solution:
        Solution1.plot_sensors()
        Solution2.plot_sensors()


if __name__ == "__main__":
    solve_with_vns(plot_solution=True)
