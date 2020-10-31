import time
import copy

def v(solution, k = 0):
    if k == 0:
        solution.voisinage2(nb_removed = 6)
    elif k == 1:
        solution.voisinage1(max_coverage = 0, q=1, nb_removed=1)
    elif k == 2:
        solution.voisinage1(max_coverage = 0, q=0, nb_removed=1)
    else:
        raise ValueError(k)

    return solution

def start_vns(solution, k_max = 3, max_time=600):
    print("----- Starting VNS ! -----")
    scores = []
    start_time = time.time()
    while (time.time() - start_time) < max_time:
        k = 0
        while k < k_max:
            solution_prim = v(copy.deepcopy(solution), k)
            solution_prim.optimize_locally()
            scores.append(solution_prim.score())
            print("VNS score: {}\nk: {}\n".format(scores[-1], k))
            if scores[-1] < solution.score():
                solution = solution_prim
                k = 0
            else:
                k += 1
    return solution, scores