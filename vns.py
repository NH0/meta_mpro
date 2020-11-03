import logging
import random
import time
import math
import copy


def find_t0(solution0, pi=0.3, number_of_neighbors=20):
    """
    Fonction pour trouver T0 la température initiale du recuit simulé
    à partir d'une proportion de solutions à accepter pi.
    Le nombre de voisins représente la taille de l'échantillon utilisé pour déterminer T0.
    """
    score0 = solution0.score
    medium_score = 0
    for i in range(number_of_neighbors):
        neighbor = copy.deepcopy(solution0)
        neighbor.neighborhood_3(to_add=int(neighbor.sensors.number_of_nodes() / 12))
        medium_score += int(neighbor.score)
    medium_score = medium_score / number_of_neighbors
    delta_f = medium_score - score0

    return - delta_f / math.log(pi)


def v(solution, k=0):
    """
    Renvoie un élément x du voisinage V(k).
    """
    if k == 0:
        solution.neighborhood_3(to_add=int(solution.score / 6))
    elif k == 1:
        solution.neighborhood_3(to_add=int(solution.score / 5))
    elif k == 2:
        solution.neighborhood_3(to_add=int(solution.score / 4))
    else:
        raise ValueError(k)

    return solution


def start_vns(solution, k_max=3, max_time=500, max_unimproving_iters=50, phi=0.8, steps=15):
    """
    Algorithme de VNS avec recuit.
    solution est la solution initiale.
    k_max est le nombre de voisinages.
    max_time est le temps max en secondes de l'algorithme.
    max_unimproving_iters est le nombre maximal d'itérations
        sans amélioration de la meilleure solution avant arrêt de l'algorithme.
    phi est le multiplicateur de la température pour le recuit.
    step est le nombre de pas ! sans amélioration ! avant de diminuer la température du recuit.
    """
    best_solution = copy.deepcopy(solution)
    current_solution = copy.deepcopy(solution)
    t0 = find_t0(solution)
    temperature = t0
    print("----- Starting VNS ! -----")
    logging.debug("Temperature_0 {}".format(t0))
    scores = []
    start_time = time.time()
    iteration = 0
    unimproving_iterations = 0
    while (time.time() - start_time) < max_time and unimproving_iterations < max_unimproving_iters:
        k = 0
        print("Current VNS score : {}".format(best_solution.score))
        logging.info("Starting new VNS loop {}\t"
                     "Temperature is {:.2e}\t"
                     "Execution time is {:.2f}\t"
                     "Not increased best for {} interations".format(
                         iteration,
                         temperature,
                         time.time() - start_time,
                         unimproving_iterations))
        while k < k_max:
            # Nouvelle solution issue d'un voisinage V(k)
            solution_prim = v(copy.deepcopy(current_solution), k)
            # Optimisation locale
            solution_prim.optimize_locally()
            # Sauvegarde du score de la solution
            scores.append(solution_prim.score)

            logging.info("VNS score: {}\tCurrent score was {}"
                         "\tBest score is {}\tk: {}\tTime {:.1f}".format(
                             scores[-1],
                             current_solution.score,
                             best_solution.score,
                             k,
                             time.time() - start_time))
            logging.debug("Scores {}".format(scores))
            # Si le dernier score est le meilleur
            if scores[-1] < best_solution.score:
                best_solution = copy.deepcopy(solution_prim)
                unimproving_iterations = 0
            else:
                unimproving_iterations += 1

            # Recuit simulé, on peut quand même garder en solution courante une
            # solution pire
            delta_f = scores[-1] - current_solution.score
            if delta_f < 0:
                current_solution = solution_prim
                k = 0
            else:
                if temperature > 1e-4:  # Evite de diviser par zéro.
                    proba = math.exp(- delta_f / temperature)
                    logging.debug("Proba {}".format(proba))
                    if random.random() < proba and delta_f > 0:
                        current_solution = solution_prim
                        k = 0
                    else:
                        k += 1
                else:
                    k += 1

            iteration += 1
            # Réduction de la température lorsque la meilleure solution n'a pas été
            # améliorée pour "steps" itérations (le +1 pour éviter une égalité au début).
            if (unimproving_iterations + 1) % steps == 0:
                logging.info("Reducing temperature {}".format(temperature))
                temperature = phi * temperature
                logging.info("Reorganizing")
                # On en profite pour réassigner la meilleure solution à la solution courante
                current_solution = copy.deepcopy(best_solution)
                # et faire une exploration avec re_organize (présenté dans le rapport)
                current_solution.re_organize(int(current_solution.score / 2), multiproc=False)

    return best_solution, scores


if __name__ == "__main__":
    print("Start from main.py")
