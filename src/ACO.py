# implementation of baseline from https://doi.org/10.1007/978-981-13-0860-4_19
from src.baseline import Baseline
import numpy as np


class ACO(Baseline):
    def __init__(self, depots, cities, stations, graph, distance, full, limit):
        super().__init__(depots, cities, stations, graph, distance, full, limit, 'Ant Colony Optimization')
        self.group = None
        self.savings = []
        self.taus = []
        self.alpha = 1
        self.beta = 2
        self.phi = 0.95
        self.ant_num = 100
        self.iterations = 10

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

    def saving_heuristics(self, depot, city_1, city_2):
        # if depot == city_1 or depot == city_2:
        #     return self.distance[city_1][city_2]
        # else:
        #     saving = self.distance[depot][city_1] + self.distance[depot][city_2] - self.distance[city_1][city_2]
        #     return saving if saving > 0 else self.distance[city_1][city_2]
        return self.distance[city_1][city_2]

    def init(self):
        # based on assignment, initialize savings and taus
        for instance in self.group:
            saving = [[self.saving_heuristics(instance[0], i, j) for j in instance] for i in instance]
            tau = [[0.01 for _ in instance] for _ in instance]
            self.savings.append(saving)
            self.taus.append(tau)

    def find_tour(self, instance, saving, tau):
        to_visit = [True for _ in instance]
        to_visit[0] = False
        current = 0
        path = [instance[current]]
        attr = [[pow(saving[i][j], self.alpha) * pow(tau[i][j], self.beta)
                 for j in range(len(instance))] for i in range(len(instance))]
        while len(path) < len(instance):
            prob = np.array([attr[current][i] if to_visit[i] else 0 for i in range(len(instance))])
            prob = prob / prob.sum()
            current = np.random.choice([i for i in range(len(instance))], p=prob)
            to_visit[current] = False
            path.append(instance[current])
        path.append(path[0])
        return path

    def update(self, instance, paths, index):
        tau = self.taus[index]
        for i in range(len(instance)):
            for j in range(len(instance)):
                tau[i][j] = tau[i][j] * self.phi
        for path in paths:
            if len(path) == 2:
                continue
            for start, end in zip(path[:-1], path[1:]):
                i = instance.index(start)
                j = instance.index(end)
                tau[i][j] += 1 / self.distance[start][end]

    def ant_colony(self, instance):
        index = self.group.index(instance)
        paths = []
        for _ in range(self.ant_num):
            paths.append(self.find_tour(instance, self.savings[index], self.taus[index]))
        self.update(instance, paths, index)
        return self.build_routes(paths)

    def solve(self):
        self.assign()
        self.init()
        for _ in range(self.iterations):
            for i in range(len(self.group)):
                routes = self.ant_colony(self.group[i])
                if len(self.group) > len(self.solution):
                    if len(routes) > 0:
                        self.solution.append(sorted(routes, key=self.tour_length)[0])
                    else:
                        self.solution.append([])
                else:
                    self.solution[i] = sorted(routes + [self.solution[i]], key=self.tour_length)[0]
        if not self.is_feasible(self.solution):
            self.solution = []
        return self.solution

    def solve_aco_only(self):
        # the original method is a hybrid aco method, now write an aco for overall problem
        pass

