import time
import copy
import logging

def v(solution, k = 0):
    if k == 0:
        solution.voisinage2(nb_removed = 20)
    elif k == 1:
        solution.voisinage1(max_coverage = 0, q=1, nb_removed=1)
    elif k == 2:
        solution.voisinage1(max_coverage = 0, q=0, nb_removed=1)
    elif k == 3:
        solution.voisinage2(nb_removed = 10)
    elif k == 4:
        solution.voisinage1(max_coverage = 0, q=2, nb_removed=1)
    elif k == 5:
        solution.voisinage1(max_coverage = 0, q=3, nb_removed=1)
    elif k == 6:
        solution.voisinage1(max_coverage = 0, q=0, nb_removed=2)
    else:
        raise ValueError(k)

    return solution

def start_vns(solution, k_max = 4, max_time=3600):
    best_solution = copy.deepcopy(solution)
    print("----- Starting VNS ! -----")
    scores = []
    start_time = time.time()
    while (time.time() - start_time) < max_time:
        k = 0
        logging.info("Starting new VNS loop")
        while k < k_max:
            solution_prim = v(copy.deepcopy(best_solution), k)
            solution_prim.optimize_locally()
            scores.append(solution_prim.score())
            logging.info("VNS score: {}\tBest score is {}\tk: {}".format(scores[-1],best_solution.score(), k))
            if scores[-1] < best_solution.score():
                best_solution = copy.deepcopy(solution_prim)
                k = 0
            else:
                k += 1
    return best_solution, scores