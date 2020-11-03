import logging
import random
import time
import math


def find_t0(solution0, pi=0.6, number_of_neighbors=10):
    score0 = solution0.score
    medium_score = 0
    for i in range(number_of_neighbors):
        neighbor = solution0.copy()
        neighbor.neighborhood_5(to_add=int(neighbor.sensors.number_of_nodes() / 12))
        medium_score += neighbor.score
    medium_score = medium_score / number_of_neighbors
    delta_f = medium_score - score0

    return - delta_f / math.log(pi)


def v(solution, k=0):
    if k == 0:
        solution.neighborhood_5(to_add=int(solution.sensors.number_of_nodes() / 10))
    elif k == 1:
        solution.neighborhood_3(to_add=int(solution.sensors.number_of_nodes() / 10))
    elif k == 2:
        solution.neighborhood_2(to_remove=int(solution.sensors.number_of_nodes() / 6))
    elif k == 3:
        solution.neighborhood_1(max_coverage=0, q=0, nb_removed=1)
    elif k == 4:
        solution.neighborhood_1(max_coverage=0, q=1, nb_removed=1)
    elif k == 5:
        solution.neighborhood_1(max_coverage=0, q=2, nb_removed=1)
    elif k == 6:
        solution.neighborhood_1(max_coverage=0, q=0, nb_removed=2)
    elif k == 7:
        solution.neighborhood_1(max_coverage=0, q=1, nb_removed=2)
    else:
        raise ValueError(k)

    return solution


def start_vns(solution, k_max=3, max_time=600, mu=0.9, steps=15):
    best_solution = solution.copy()
    current_solution = solution.copy()
    t0 = find_t0(solution)
    temperature = t0
    print("----- Starting VNS ! -----")
    logging.debug("Temperature_0 {}".format(t0))
    scores = []
    start_time = time.time()
    iteration = 0
    while (time.time() - start_time) < max_time:
        k = 0
        print("Current VNS score : {}".format(best_solution.score))
        logging.info("Starting new VNS loop\tExecution time is {:.2f}".format(
            time.time() - start_time))
        while k < k_max:
            solution_prim = v(current_solution.copy(), k)
            solution_prim.optimize_locally()
            scores.append(solution_prim.score)
            logging.info("VNS score: {}\tCurrent score was {}\tBest score is {}\tk: {}".format(
                scores[-1],
                current_solution.score,
                best_solution.score,
                k))
            logging.debug("Scores {}\tTemperature {}".format(scores, temperature))
            if scores[-1] < best_solution.score:
                best_solution = solution_prim.copy()
            if scores[-1] < current_solution.score:
                current_solution = solution_prim.copy()
                k = 0
            else:
                delta_f = scores[-1] - best_solution.score
                proba = math.exp(- delta_f / temperature)
                logging.debug("Proba {}".format(proba))
                if random.random() < proba:
                    current_solution = solution_prim
                    if iteration % steps == 0:
                        temperature = mu * temperature
                    k = 0
                else:
                    k += 1
        iteration += 1

    return best_solution, scores
