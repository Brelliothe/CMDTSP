# https://doi.org/10.1016/j.tre.2021.102293

from baseline import Baseline
import copy
import numpy as np
import time


class VNS(Baseline):
    def __init__(self, depots, cities, stations, graph, distance, full, limit):
        super().__init__(depots, cities, stations, graph, distance, full, limit, 'Variable Neighborhood Search')
        # Now depots, cities, stations are node names
        self.group = None
        self.num_ns = 5  # 5 neighborhood structure: 1-0 move, 1-1 exchange, 2-2 exchange, 1-2 exchange, 1-1-1 exchange
        self.num_ts = 4  # 4 neighborhood structure: 1-add-station, 1-drop-station, 1-swap-station, 2-opt
        self.theta = 20  # maximum allowed iterations for solution not improved
        self.iters = 100  # maximum iterations allowed
        self.alpha = 0.5
        self.beta = 0.5
        self.delta = 0.5
        self.time = 1000
        self.cost = None
        self.solutions = []

    def assign(self):
        self.group = [[depot] for depot in self.depots]
        for city in self.cities:
            nearby = 1000000
            source = None
            for i in range(len(self.depots)):
                if self.distance[self.depots[i]][city] < nearby:
                    nearby = self.distance[self.depots[i]][city]
                    source = i
            assert source is not None, "No closest depot, impossible"
            self.group[source].append(city)

    def initial_solution(self):
        self.assign()
        self.solution = []
        for group in self.group:
            self.solution.append(self.lkh(group))
        self.cost = self.solution_cost(self.solution)
        self.solutions.append(copy.deepcopy(self.solution))

    def tabu_search(self, solution, i):
        if i == 0:
            min_cost = 10000000
            best_place = None
            for j in range(len(solution)):
                for k in range(j, len(solution)):
                    tour_1 = solution[j]
                    tour_2 = solution[k]
                    for l in range(1, len(tour_1) - 1):
                        for m in range(1, len(tour_2) - 1):
                            tmp = tour_1[l]
                            tour_1[l] = tour_2[m]
                            tour_2[m] = tmp
                            if self.solution_cost(solution) < min_cost:
                                min_cost = self.solution_cost(solution)
                                best_place = [j, k, copy.deepcopy(tour_1), copy.deepcopy(tour_2)]
                            tour_2[m] = tour_1[l]
                            tour_1[l] = tmp
            if best_place is not None:
                solution[best_place[0]] = best_place[2]
                solution[best_place[1]] = best_place[3]
        elif i == 1:
            min_cost = 1000000
            best_place = None
            for station in self.stations:
                # if sum([count_num(tour, station) for tour in solution]) < self.limits:
                if sum([tour.count(station) for tour in solution]) < self.limit:
                    for j in range(len(solution)):
                        tour = solution[j]
                        for k in range(1, len(tour)):
                            tour.insert(k, station)
                            if self.solution_cost(solution) < min_cost:
                                min_cost = self.solution_cost(solution)
                                best_place = [j, copy.deepcopy(tour)]
                            tour.pop(k)
            if best_place is not None:
                solution[best_place[0]] = best_place[1]
        elif i == 2:
            min_cost = 1000000
            best_place = None
            for k in range(len(solution)):
                tour = solution[k]
                for j in range(len(tour)):
                    # if get_index(self.stations, tour[j]) > -1:
                    if tour[j] in self.stations:
                        station = tour.pop(j)
                        if self.solution_cost(solution) < min_cost:
                            min_cost = self.solution_cost(solution)
                            best_place = [k, copy.deepcopy(tour)]
                        tour.insert(j, station)
            if best_place is not None:
                solution[best_place[0]] = best_place[1]
        elif i == 3:
            solution = self.tabu_search(solution, 1)
            solution = self.tabu_search(solution, 2)
        else:
            assert 0 == 1, "tabu structure not implemented"
        return solution

    def shake(self, i):
        # to check the existence of 2 consecutive customers
        def exists(path):
            for j in range(len(path) - 1):
                # if get_index(self.cities, path[j]) > -1 and get_index(self.cities, path[j + 1]) > -1:
                if path[j] in self.cities and path[j + 1] in self.cities:
                    return True
            return False

        solution = copy.deepcopy(self.solution)
        if i == 0:
            city = self.cities[np.random.choice([i for i in range(len(self.cities))], 1).item()]
            tour = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()]
            for sol in solution:
                if city in sol:
                    sol.remove(city)
            index = np.random.choice([i for i in range(1, len(tour))], 1).item()
            tour.insert(index, city)
        elif i == 1:
            indices = np.random.choice([i for i in range(len(self.cities))], 2, replace=False)
            city_1, city_2 = self.cities[indices[0]], self.cities[indices[1]]
            for sol in solution:
                # index_1 = get_index(sol, city_1)
                # index_2 = get_index(sol, city_2)
                index_1 = sol.index(city_1) if city_1 in sol else -1
                index_2 = sol.index(city_2) if city_2 in sol else -1
                if index_1 > -1:
                    sol[index_1] = city_2
                if index_2 > -1:
                    sol[index_2] = city_1
        elif i == 2:
            # TODO: if no consecutive customers
            tour_1 = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()]
            while not exists(tour_1):
                tour_1 = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()]
            tour_2 = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()]
            while not exists(tour_2):
                tour_2 = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()]
            swap_1, swap_2 = None, None
            start_1, start_2 = None, None
            while swap_1 is None or swap_2 is None:
                start_1 = np.random.choice([i for i in range(len(tour_1) - 1)], 1).item() if swap_1 is None else start_1
                start_2 = np.random.choice([i for i in range(len(tour_2) - 1)], 1).item() if swap_2 is None else start_2
                if tour_1[start_1] in self.cities:
                    end_1 = start_1 + 1
                    while end_1 < len(tour_1) and tour_1[end_1] not in self.cities:
                        end_1 += 1
                    if end_1 < len(tour_1):
                        swap_1 = copy.deepcopy(tour_1[start_1: end_1 + 1])
                if tour_2[start_2] in self.cities:
                    end_2 = start_2 + 1
                    while end_2 < len(tour_2) and tour_2[end_2] not in self.cities:
                        end_2 += 1
                    if end_2 < len(tour_2):
                        swap_2 = copy.deepcopy(tour_2[start_2: end_2 + 1])
            tour_1 = tour_1[:start_1] + swap_2 + tour_1[end_1 + 1:]
            tour_2 = tour_2[:start_2] + swap_1 + tour_2[end_2 + 1:]
        elif i == 3:
            tour_1, tour_2 = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()], None
            while not exists(tour_1):
                tour_1 = solution[np.random.choice([i for i in range(len(self.depots))], 1).item()]
            swap = None
            start, end = None, None
            while swap is None:
                start = np.random.choice([i for i in range(len(tour_1) - 1)], 1).item()
                # if get_index(self.cities, tour_1[start]) > -1:
                if tour_1[start] in self.cities:
                    end = start + 1
                    # while end < len(tour_1) and get_index(self.cities, tour_1[end]) == -1:
                    while end < len(tour_1) and tour_1[end] not in self.cities:
                        end += 1
                    if end < len(tour_1):
                        swap = tour_1[start: end + 1]
            indices = [i for i in range(len(self.cities))]
            indices.remove(np.where(self.cities == swap[0])[0].item())
            indices.remove(np.where(self.cities == swap[-1])[0].item())
            city = self.cities[np.random.choice(indices, 1).item()]
            for tour in solution:
                if city in tour:
                    tour_2 = copy.deepcopy(tour)
            index = tour_2.index(city)
            tour_1 = tour_1[:start] + [tour_2[index]] + tour_1[end + 1:]
            tour_2 = tour_2[:index] + swap + tour_2[index + 1:]
        elif i == 4:
            try:
                indices = np.random.choice([i for i in range(len(self.depots)) if len(solution[i]) > 2], 3)
                swap = [np.random.choice([i for i in range(len(solution[index]))
                                          if solution[index][i] in self.cities], 1).item() for index in indices]
                temp = solution[indices[0]][swap[0]]
                solution[indices[0]][swap[0]] = solution[indices[1]][swap[1]]
                solution[indices[1]][swap[1]] = solution[indices[2]][swap[2]]
                solution[indices[2]][swap[2]] = temp
            except Exception as e:
                print('exception: ', e)
            finally:
                pass
        return solution

    def solution_cost(self, solution, indicate=False):
        cost = 0
        for tour in solution:
            length = self.tour_length(tour)
            energy = self.full
            for start, end in zip(tour[:-1], tour[1:]):
                energy -= self.distance[start][end]
                if energy < 0:
                    cost += - energy * self.alpha
                    if indicate:
                        print(start, end)
                if end in self.stations:
                    energy = self.full
            cost += length + self.beta * (length - self.time) if length > self.time else length
        return cost

    def solve(self):
        self.initial_solution()
        total_rounds, non_imp_rounds, run_time = 0, 0, 0
        while non_imp_rounds < self.theta and total_rounds < self.iters and run_time < self.run_time:
            total_rounds += 1
            non_imp_rounds += 1
            ns, ts = 0, 0
            while ns < self.num_ns + 1 and run_time < self.run_time:
                shake_time = time.time()
                solution = self.shake(ns)
                run_time += time.time() - shake_time
                while ts < self.num_ts and run_time < self.run_time:
                    search_time = time.time()
                    solution = self.tabu_search(solution, ts)
                    if solution not in self.solutions and \
                            self.solution_cost(solution) < self.solution_cost(self.solution):
                        self.solution = copy.deepcopy(solution)
                        self.solutions.append(copy.deepcopy(solution))
                        ns = -1
                        ts = 0
                        non_imp_rounds = 0
                        self.alpha = self.alpha / (1 + self.delta)
                        self.beta = self.beta / (1 + self.delta)
                        self.cost = self.solution_cost(self.solution)
                    else:
                        ts += 1
                        self.alpha = self.alpha * (1 + self.delta)
                        self.beta = self.beta * (1 + self.delta)
                        self.cost = self.solution_cost(self.solution)
                    run_time += time.time() - search_time
                ns += 1
