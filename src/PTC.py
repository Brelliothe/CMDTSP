# our method

from baseline import Baseline
import gurobipy as gp
from gurobipy import GRB
from itertools import islice
import networkx as nx
import numpy as np


class PTC(Baseline):
    def __init__(self, depots, cities, stations, graph, distance, full, limit):
        super().__init__(depots, cities, stations, graph, distance, full, limit, 'Hierarchical Pipeline')
        self.k = 10
        self.group = []
        self.lb, self.ub = 0, 1000
        self.shortest_paths = {}

    def partition(self):
        """
        allocate the cities to salesman by partition the minimum spanning tree
        :return: (stations, depots, cities) with matched depots and cities
        """
        # find the minimum spanning tree
        graph = nx.Graph()
        for i in np.concatenate((self.depots, self.cities)):
            graph.add_node(i)
        for i in graph.nodes:
            for j in graph.nodes:
                if i == j:
                    continue
                graph.add_edge(i, j, weight=self.distance[i][j])
        tree = nx.minimum_spanning_tree(graph)

        # turn the spanning tree into a rooted tree
        for i in tree.nodes:
            tree.nodes[i]['parent'] = -1

        def rooted_tree(node):
            # con represent the minimum cost in the subtree when the connected depot is its child
            # ncon represent the minimum cost in the subtree when the connected depot is not its child
            # for the depot node, treat itself as its child, set its ncon as -1 to be invalid
            cons, ncons, index, diff = [], [], [], []
            for n in tree.neighbors(node):
                if n != tree.nodes[node]['parent']:
                    # indicator whether n will connect to a depot via its parent
                    tree.nodes[n]['pcon'] = True
                    tree.nodes[n]['parent'] = node
                    con, ncon = rooted_tree(n)
                    tree.nodes[n]['con'] = con
                    tree.nodes[n]['ncon'] = ncon
                    # n is a source
                    if n in self.depots:
                        cons.append(con + tree[node][n]['weight'])
                        ncons.append(con)
                        diff.append(cons[-1] - ncons[-1])
                        index.append(n)
                    # n is a target and has source inside the subtree
                    elif con != -1:
                        cons.append(con + tree[node][n]['weight'])
                        ncons.append(min(ncon + tree[node][n]['weight'], con))
                        # the node will not connect to its parent if not connect source to its parent
                        diff.append(cons[-1] - ncons[-1])
                        index.append(n)
                        if ncons[-1] == con:
                            tree.nodes[n]['pcon'] = False
                    # n do not have source within the subtree
                    else:
                        ncons.append(ncon + tree[node][n]['weight'])
            if node in self.depots:
                # it is automatically connected to itself
                con = sum(ncons)
                ncon = -1
            elif len(diff) == 0:
                con = -1
                ncon = sum(ncons)
            else:
                id = np.argmin(np.array(diff))
                con = min(diff) + sum(ncons)
                ncon = sum(ncons)
                tree.nodes[node]['child'] = index[id]
            return con, ncon

        con, ncon = rooted_tree(self.depots[0])
        tree.nodes[self.depots[0]]['con'] = con
        tree.nodes[self.depots[0]]['ncon'] = ncon
        tree.nodes[self.depots[0]]['pcon'] = False

        def assign_group(node, value):
            if node in self.depots:
                # case 1: node is a depot, all children do not connect to both it and a depot inside the subtree
                # the node belongs to the group named by itself
                tree.nodes[node]['group'] = np.where(self.depots == node)[0].item()
                # for all neighbor nodes besides the parent
                for n in tree.neighbors(node):
                    if n != tree.nodes[node]['parent']:
                        # if node n is also a depot
                        if n in self.depots:
                            # it belongs to the group named by itself
                            tree.nodes[n]['group'] = np.where(self.depots == n)[0].item()
                            assign_group(n, tree.nodes[n]['con'])
                        # if node n is a city but connects to 'node'
                        elif tree.nodes[n]['pcon']:
                            # assign its group to be node
                            tree.nodes[n]['group'] = tree.nodes[node]['group']
                            assign_group(n, tree.nodes[n]['ncon'])
                        # node n is a city and connects to some depot inside the subtree
                        else:
                            tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])

            # case 2: node is a city, and it connects to a depot inside the subtree
            elif value == tree.nodes[node]['con']:
                # find its child whose subtree contains the depot
                n = tree.nodes[node]['child']
                # if child n is a depot
                if n in self.depots:
                    index = np.where(self.depots == n)[0].item()
                    tree.nodes[node]['group'] = index
                    tree.nodes[n]['group'] = index
                    assign_group(n, tree.nodes[n]['con'])
                # if child n is a city
                else:
                    tree.nodes[node]['group'] = assign_group(n, tree.nodes[n]['con'])
                # other neighbors besides child n and parent
                for n in tree.neighbors(node):
                    if n != tree.nodes[node]['parent'] and n != tree.nodes[node]['child']:
                        # depot should not connect to it
                        if n in self.depots:
                            tree.nodes[n]['group'] = np.where(self.depots == n)[0].item()
                            assign_group(n, tree.nodes[n]['con'])
                        # city connect to 'node'
                        elif tree.nodes[n]['pcon']:
                            tree.nodes[n]['group'] = tree.nodes[node]['group']
                            assign_group(n, tree.nodes[n]['ncon'])
                        # city not connect to 'node'
                        else:
                            tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])

            else:
                # node is a city and the connected depot is outside the subtree
                # a city is visited before so node has already been assigned a group
                for n in tree.neighbors(node):
                    if n != tree.nodes[node]['parent']:
                        if n in self.depots:
                            tree.nodes[n]['group'] = np.where(self.depots == n)[0].item()
                            assign_group(n, tree.nodes[n]['con'])
                        elif tree.nodes[n]['pcon']:
                            tree.nodes[n]['group'] = tree.nodes[node]['group']
                            assign_group(n, tree.nodes[n]['ncon'])
                        else:
                            tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])
            return tree.nodes[node]['group']

        # start from the first node
        assign_group(self.depots[0], tree.nodes[self.depots[0]]['con'])

        self.group = [[depot] for depot in self.depots]
        for node in tree.nodes:
            if node not in self.depots:
                self.group[tree.nodes[node]['group']].append(node)

    def get_shortest_paths(self):
        graph = nx.Graph()
        for i in self.stations:
            graph.add_node(i)
        for i in graph.nodes:
            for j in graph.nodes:
                if i == j:
                    continue
                graph.add_edge(i, j, weight=self.distance[i][j])
        for i in self.stations:
            for j in self.stations:
                if i != j:
                    self.shortest_paths[i, j] = [self.k_shortest_paths(graph, i, j, self.k), []]
                    for path in self.shortest_paths[i, j][0]:
                        self.shortest_paths[i, j][-1].append(self.path_length(graph, path))
                else:
                    self.shortest_paths[i, i] = [[[i] for _ in range(self.k)], [0 for _ in range(self.k)]]

    @staticmethod
    def k_shortest_paths(graph, i, j, k):
        return list(islice(nx.shortest_simple_paths(graph, i, j, weight='weight'), k))

    @staticmethod
    def path_length(graph, path):
        length = 0
        for u, v in zip(path[:-1], path[1:]):
            length += graph[u][v]['weight']
        return length

    def heuristic_search(self, visit):
        paths, costs, energy = [], [], []

        # for each pair of city, find the shortest path
        for source, target in zip(visit[:-1], visit[1:]):
            paths.append([])
            costs.append([])
            energy.append([])
            if self.distance[source][target] <= self.full:
                paths[-1].append([source, target])
                costs[-1].append(self.distance[source][target])
                energy[-1].append((0, 1e10, self.distance[source][target]))
            path_candidates = []

            # check all valid paths
            for i in self.stations:
                for j in self.stations:
                    for k in range(len(self.shortest_paths[i, j][0])):
                        path_candidates.append((i, j, k, self.shortest_paths[i, j][-1][k]
                                                + self.distance[i][source]
                                                + self.distance[i][target]))
            # sort valid paths by the cost
            path_candidates.sort(key=lambda t: t[-1])
            # take the top num paths, fill empty part by the last valid path
            while len(paths[-1]) < self.k:
                if len(path_candidates) == 0:
                    paths[-1].append(paths[-1][-1])
                    costs[-1].append(costs[-1][-1])
                    energy[-1].append(energy[-1][-1])
                    continue
                i, j, k, c = path_candidates.pop(0)
                path = [source] + [t for t in self.shortest_paths[i, j][0][k]] + [target]
                if path not in paths[-1]:
                    paths[-1].append(path)
                    costs[-1].append(c)
                    energy[-1].append((self.distance[source][path[1]], self.full - self.distance[target][path[-2]], -1e10))
        return paths, costs, energy

    def congestion_control(self, paths, costs, power):
        solution, cost = [], 0
        a, d, n, s = len(paths), max([len(x) for x in paths]), len(paths[0][0]), len(paths[0][0][0])
        model = gp.Model('control')
        model.setParam(GRB.Param.OutputFlag, 0)
        choices = model.addVars(a, d, n, vtype=GRB.BINARY, name='choices')
        energy = model.addVars(a, d + 1, vtype=GRB.CONTINUOUS, name='energy')
        model.addConstrs(choices[i, j, k] == 0 for i in range(a) for j in range(len(paths[i]), d) for k in range(n))
        model.addConstrs(gp.quicksum([choices[i, j, k] for k in range(n)]) == 1 for i in range(a)
                         for j in range(len(paths[i])))
        model.addConstrs(gp.quicksum([choices[i, j, k] * paths[i][j][k][m] for i in range(a)
                                      for j in range(len(paths[i])) for k in range(n)]) <= self.limit for m in range(s))
        model.addConstrs(energy[i, j] >= 0 for i in range(a) for j in range(d + 1))
        model.addConstrs(
            energy[i, j] >= gp.quicksum([power[i][j][k][0] * choices[i, j, k] for k in range(n)]) for i in range(a)
            for j in range(len(paths[i])))
        model.addConstrs(
            energy[i, j] <= gp.quicksum([power[i][j - 1][k][1] * choices[i, j - 1, k] for k in range(n)])
            for i in range(a) for j in range(1, len(paths[i]) + 1))
        model.addConstrs(
            energy[i, j] - energy[i, j + 1] >= gp.quicksum([power[i][j][k][2] * choices[i, j, k] for k in range(n)])
            for i in range(a) for j in range(len(paths[i])))
        model.addConstrs(energy[i, j] <= self.full for j in range(d + 1) for i in range(a))
        model.addConstrs(energy[i, j] == 0 for i in range(a) for j in range(len(paths[i]) + 1, d + 1))
        model.setObjective(
            gp.quicksum([choices[i, j, k] * costs[i][j][k] for i in range(a) for j in range(len(paths[i]))
                         for k in range(n)]))
        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("model.ilp")
            return None, 0
        else:
            for i in range(a):
                solution.append([])
                for j in range(d):
                    for k in range(n):
                        if choices[i, j, k].x > 0.5:
                            solution[i].append((j, k))
                            cost += costs[i][j][k]
        return solution, cost

    def solve(self):
        self.partition()
        self.get_shortest_paths()
        paths, proposal, price, power = [], [], [], []
        for group in self.group:
            visit = self.lkh(group)
            path, cost, energy = self.heuristic_search(visit)
            paths.append(path)
            proposal.append([[[1 if s in path[i][j] else 0 for s in self.stations] for j in range(self.k)] for i in
                             range(len(path))])
            price.append(cost)
            power.append(energy)
        solution, cost = self.congestion_control(proposal, price, power)
        if solution is not None:
            solution = [[loc for index in solution[i] for loc in paths[i][index[0]][index[1]][:-1]]
                        for i in range(len(solution))]
            self.solution = [solution[i] + [self.depots[i]] for i in range(len(solution))]
