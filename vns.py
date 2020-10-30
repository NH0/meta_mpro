import time
import copy

def v(solution, k = 0):
    if k == 0:
        solution.voisinage1(max_coverage = 0, q=0, nb_removed=1)
    elif k == 1:
        solution.voisinage1(max_coverage = 0, q=1, nb_removed=1)
    elif k == 2:
        solution.voisinage2(nb_removed = 4)
    else:
        raise ValueError(k)

    return solution

def start_vns(solution, k_max = 3, max_time=300):
    print("###### Starting VNS !")
    start_time = time.time()
    while (time.time() - start_time) < max_time:
        k = 1
        while k <= k_max:
            solution_prim = v(copy.deepcopy(solution), k)
            solution_prim.optimize_locally()
            if solution_prim.score() < solution.score():
                solution = solution_prim
                k = 1
            else:
                k += 1
    return solution