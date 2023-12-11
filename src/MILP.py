from baseline import Baseline
import gurobipy as gp
from gurobipy import GRB
import numpy as np


class MILP(Baseline):
    def __init__(self, depots, cities, stations, graph, distance, full, limit):
        super().__init__(depots, cities, stations, graph, distance, full, limit, 'Mixture Integer Linear Programming')

    def solve(self):
        # construct the MILP
        agents = len(self.depots)
        goals = len(self.depots) + len(self.cities)
        locs = len(self.depots) + len(self.cities) + len(self.stations)
        shift = len(self.stations)
        locations = np.concatenate((self.stations, self.depots, self.cities))
        distance = [[] for _ in locations]
        full, limit = self.full, self.limit
        for i in range(len(locations)):
            for j in range(len(locations)):
                distance[i].append(self.distance[locations[i]][locations[j]])

        model = gp.Model('solution')

        # do not output intermediate results
        model.setParam(GRB.Param.OutputFlag, 0)

        # set time limit
        model.setParam('Timelimit', 600)

        # binary variable representing whether salesman from depot i visit city j, shape (agents, goals)
        group = model.addVars(agents, goals, vtype=GRB.BINARY, name='group')

        # each city is visited exactly once by exactly one agent
        model.addConstrs(gp.quicksum([group[a, g] for a in range(agents)]) == 1 for g in range(goals))

        # allocate depots, i.e, source[a] belongs to group[a]
        model.addConstrs(group[a, a] == 1 for a in range(agents))

        # traveling salesman requirements (Gavish-Graves formulation)
        # selection of high level edge, uniquely determine the tsp solution
        tsp = model.addVars(agents, goals, goals, vtype=GRB.BINARY, name='tsp')

        # in-degree equals to 1
        model.addConstrs(gp.quicksum([tsp[a, i, j] for i in range(goals)]) == group[a, j] for a in range(agents)
                         for j in range(goals) if j != a)
        # out-degree equals to 1
        model.addConstrs(gp.quicksum([tsp[a, i, j] for j in range(goals)]) == group[a, i] for a in range(agents)
                         for i in range(goals) if i != a)
        # consider the case that the salesman do not need to visit any city, which should only have group[a, a] == 1
        model.addConstrs(gp.quicksum([tsp[a, i, a] for i in range(goals)]) ==
                         gp.quicksum([tsp[a, a, i] for i in range(goals)]) for a in range(agents))
        model.addConstrs(gp.quicksum([tsp[a, i, a] for i in range(goals)]) >= group[a, j] for a in range(agents)
                         for j in range(goals) if j != a)
        model.addConstrs(gp.quicksum([tsp[a, i, a] for i in range(goals)]) <= 1 for a in range(agents))
        model.addConstrs(gp.quicksum([tsp[a, i, a] for i in range(goals)]) + 1 <=
                         gp.quicksum([group[a, i] for i in range(goals)]) for a in range(agents))

        # number of items remained at each edge
        flow = model.addVars(agents, goals, goals, vtype=GRB.CONTINUOUS, name='flow')
        # carry one item for each target
        model.addConstrs(flow[a, i, j] <= goals * tsp[a, i, j] for a in range(agents) for i in range(goals)
                         for j in range(goals))
        # initial number of items, equals to the number of cities to visit
        model.addConstrs(gp.quicksum([flow[a, a, j] for j in range(goals)]) ==
                         gp.quicksum([group[a, g] for g in range(goals)]) - 1 for a in range(agents))
        # drop one item at each visited target
        model.addConstrs(gp.quicksum([flow[a, i, j] for i in range(goals)]) -
                         gp.quicksum([flow[a, j, k] for k in range(goals)]) == group[a, j] for a in range(agents)
                         for j in range(agents, goals))
        # no self loop constraint
        model.addConstrs(tsp[a, i, i] == 0 for a in range(agents) for i in range(goals))
        model.addConstrs(flow[a, i, i] == 0 for a in range(agents) for i in range(goals))
        # no edge between depots
        model.addConstrs(
            tsp[a, i, j] == 0 for a in range(agents) for i in range(goals) for j in range(agents) if j != a)
        model.addConstrs(
            flow[a, i, j] == 0 for a in range(agents) for i in range(goals) for j in range(agents) if j != a)

        # selection of low level edge, uniquely determine the whole solution
        # representing whether in problem for a salesman a, edge (k,m) is contained in the path between city i and city j
        select = model.addVars(agents, goals, goals, locs, locs, vtype=GRB.BINARY, name='select')
        # resource constraints for each station
        model.addConstrs(gp.quicksum([select[a, i, j, k, m] for a in range(agents) for i in range(goals)
                                      for j in range(goals) for m in range(locs)]) <= limit for k in range(shift))
        # out degree equals to 1 if high level path i -> j is selected
        model.addConstrs(gp.quicksum([select[a, i, j, i + shift, m] for m in range(locs)]) == tsp[a, i, j]
                         for a in range(agents) for i in range(goals) for j in range(goals))
        # in degree equals to 1 if high level path i -> j is selected
        model.addConstrs(gp.quicksum([select[a, i, j, k, j + shift] for k in range(locs)]) == tsp[a, i, j]
                         for a in range(agents) for i in range(goals) for j in range(goals))

        # out degree no more than 1
        model.addConstrs(gp.quicksum([select[a, i, j, k, m] for m in range(locs)]) <= 1 for a in range(agents)
                         for i in range(goals) for j in range(goals) for k in range(locs))
        # in degree no more than 1
        model.addConstrs(gp.quicksum([select[a, i, j, k, m] for k in range(locs)]) <= 1 for a in range(agents)
                         for i in range(goals) for j in range(goals) for m in range(locs))
        # in degree equals to out degree
        model.addConstrs(gp.quicksum([select[a, i, j, k, m] for m in range(locs)]) ==
                         gp.quicksum([select[a, i, j, m, k] for m in range(locs)])
                         for a in range(agents) for i in range(goals) for j in range(goals)
                         for k in range(locs) if k != i + shift and k != j + shift)

        # i has no in degree and j has no out degree to force to select the path i -> j
        model.addConstrs(gp.quicksum([select[a, i, j, k, i + shift] for k in range(locs)]) == 0
                         for a in range(agents) for i in range(goals) for j in range(goals))
        model.addConstrs(gp.quicksum([select[a, i, j, j + shift, m] for m in range(locs)]) == 0
                         for a in range(agents) for i in range(goals) for j in range(goals))
        # no self loop
        model.addConstrs(select[a, i, j, k, k] == 0 for a in range(agents) for i in range(goals) for j in range(goals)
                         for k in range(locs))
        model.addConstrs(gp.quicksum([select[a, i, j, k, m] for m in range(locs)]) == 0 for a in range(agents)
                         for i in range(goals) for j in range(goals) for k in range(shift, locs)
                         if k != i + shift and k != j + shift)

        # energy constraints
        energy = model.addVars(goals, vtype=GRB.CONTINUOUS, name='energy')
        # initialize energy to be full
        model.addConstrs(energy[a] == full for a in range(agents))
        # energy should be non-negative
        model.addConstrs(energy[g] >= 0 for g in range(goals))

        # energy constraints between stations: should not choose station further than energy limit
        model.addConstrs(select[a, i, j, k, m] * distance[k][m] <= full for a in range(agents) for i in range(goals)
                         for j in range(goals) for k in range(shift) for m in range(shift))
        # energy constraints for leaving locs
        model.addConstrs(select[a, i, j, i + shift, m] * distance[i + shift][m] <= energy[i]
                         for a in range(agents) for i in range(goals) for j in range(goals) for m in range(locs))
        # energy constraints for arriving locs from stations except for the return back to depot
        model.addConstrs(energy[j] <= full - select[a, i, j, k, j + shift] * distance[k][j + shift]
                         for a in range(agents) for i in range(goals) for j in range(goals) for k in range(shift)
                         if j != a)
        # energy constraints for directly arriving a city from a city
        model.addConstrs(energy[i] - energy[j] >= select[a, i, j, i + shift, j + shift] * distance[i + shift][j + shift]
                         - (1 - select[a, i, j, i + shift, j + shift]) * full
                         for a in range(agents) for i in range(goals) for j in range(goals) if j != a)
        # for returning back to the depot
        model.addConstrs(energy[i] >= select[a, i, a, i + shift, a + shift] * distance[i + shift][a + shift] -
                         (1 - select[a, i, a, i + shift, a + shift]) * full for a in range(agents) for i in
                         range(goals))
        model.addConstrs(0 <= full - select[a, i, a, k, a + shift] * distance[k][a + shift] for a in range(agents)
                         for i in range(goals) for k in range(locs))

        # cost of the whole path
        model.setObjective(gp.quicksum([select[a, i, j, k, m] * distance[k][m] for a in range(agents)
                                        for i in range(goals) for j in range(goals) for k in range(locs)
                                        for m in range(locs)]), GRB.MINIMIZE)
        model.optimize()
        # handle the cases of infeasible instances and time out
        if model.Status == GRB.INFEASIBLE:
            print('infeasible')
            model.computeIIS()
            for c in model.getConstrs():
                if c.IISConstr:
                    print(model.getRow(c))
            self.solution = []
        # construct the solution from binary variables
        elif model.Status == GRB.TIME_LIMIT:
            print('out of time')
        else:
            solution = [[depot] for depot in self.depots]
            edges = [[] for _ in self.depots]
            loc_seq = [[] for _ in self.depots]
            # get city tours
            for key in tsp.keys():
                if tsp[key].x > 0.5:
                    edges[key[0]].append((key[1], key[2]))
            for index in range(agents):
                start = index
                loc_seq[index].append(index)
                for _ in range(len(edges[index])):
                    for edge in edges[index]:
                        if edge[0] == start:
                            end = edge[1]
                    loc_seq[index].append(end)
                    start = end
            # fill between cities
            for index in range(agents):
                for i, j in zip(loc_seq[index][:-1], loc_seq[index][1:]):
                    e = []
                    for k in range(locs):
                        for m in range(locs):
                            if select[index, i, j, k, m].x > 0.5:
                                e.append((k, m))
                    start = shift + i
                    for _ in e:
                        for edge in e:
                            if edge[0] == start:
                                end = edge[1]
                        solution[index].append(locations[end])
                        start = end
            self.solution = solution
