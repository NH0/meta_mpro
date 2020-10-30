import time

def v(solution, k = 0):
    if k == 0:
        return solution
    elif k == 1:
        return solution
    elif k == 2:
        return solution
    else:
        raise ValueError(k)

def vns(solution, k_max = 3, max_time=300):
    start_time = time.time()
    while (time.time() - start_time) < max_time:
        k = 1
        while k <= k_max:
            solution_prim = solution
            solution_prim.optimize_locally()
            if solution_prim.score() < solution.score():
                solution = solution_prim
                k = 1
            else:
                k += 1
    return solution