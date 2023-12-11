import networkx as nx
import numpy as np
import pandas as pd
import os
import osmnx as ox
import pickle
import re
from tqdm import tqdm
from src.utils import euclidean, haversine, compute_distance_on_graph


path = "datasets"


def manhattan():
    g = nx.MultiGraph(nx.read_graphml(path + '/nyc.graphml'))
    graph = nx.MultiGraph()
    index = {node: i for node, i in zip(g.nodes, range(len(g.nodes)))}
    for node in g.nodes:
        graph.add_node(index[node], pos=[float(g.nodes[node]['lon']), float(g.nodes[node]['lat'])])
    for edge in g.edges:
        graph.add_edge(index[edge[0]], index[edge[1]],
                       weight=haversine(graph.nodes[index[edge[0]]]['pos'][0], graph.nodes[index[edge[0]]]['pos'][1],
                                        graph.nodes[index[edge[1]]]['pos'][0], graph.nodes[index[edge[1]]]['pos'][1]))
    return graph


def boston():
    place_name = "Cambridge, MA, USA"
    graph = ox.graph_from_place(place_name, network_type='drive')
    graph = graph.to_undirected()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nx.set_node_attributes(graph, {node: [graph.nodes[node]['x'], graph.nodes[node]['y']] for node in graph.nodes},
                           name='pos')
    data = pd.read_csv(path + '/current_bluebikes_stations.csv', header=None, skiprows=2)
    stations = []
    for index, row in data.iterrows():
        if row.iloc[4] == 'Cambridge':
            pos = [float(row.iloc[2]), float(row.iloc[3])]
            min_distance = 100000
            min_node = None
            for node in graph.nodes:
                if haversine(graph.nodes[node]['x'], graph.nodes[node]['y'], pos[1], pos[0]) < min_distance:
                    min_distance = haversine(graph.nodes[node]['x'], graph.nodes[node]['y'], pos[1], pos[0])
                    min_node = node
            stations.append(min_node)
    return graph, stations


def md_vrp():
    graph = nx.Graph()
    pos = {}
    depots, cities = [],  []
    with open(path + '/mdvrp/pr10', 'r') as f:
        lines = f.readlines()
        line = re.sub(r'\s+', ' ', lines[0].strip())
        t, m, n, d = tuple([int(x) for x in line.split(' ')])
        assert int(t) == 2, 'wrong problem type'
        for j in range(d + 1, n + d + 1):
            info = lines[j].strip()
            info = re.sub(r'\s+', ' ', info)
            info = info.split()
            graph.add_node(j - d - 1, pos=[float(info[1]), float(info[2])], name='city')
            cities.append(j - d - 1)
            pos[j - d - 1] = (float(info[1]), float(info[2]))
        for j in range(n + d + 1, n + d * 2 + 1):
            info = lines[j].strip()
            info = re.sub(r'\s+', ' ', info)
            info = info.split()
            graph.add_node(j - d - 1, pos=[float(info[1]), float(info[2])], name='depot')
            depots.append(j - d - 1)
            pos[j - d - 1] = (float(info[1]), float(info[2]))
    for start in graph.nodes:
        for end in graph.nodes:
            graph.add_edge(start, end, weight=euclidean(graph.nodes[start]['pos'], graph.nodes[end]['pos']))
    return graph, depots, cities


def random_instance(depots, cities, stations):
    np.random.seed(0)
    locations = np.random.random((depots + cities + stations, 2))
    graph = nx.Graph()
    for i in range(depots + cities + stations):
        for j in range(i, depots + cities + stations):
            graph.add_edge(i, j, weight=np.linalg.norm(locations[i] - locations[j]))
    return graph, locations[:depots], locations[depots: depots + cities], locations[-stations:]


def random_graph_instance(graph, depots, cities, stations):
    # cities and depots can be random generated, stations should be more uniform
    nodes = np.random.choice(graph.nodes, depots + cities + stations, replace=False)
    return nodes[:depots], nodes[depots: depots + cities], nodes[depots + cities:]


def random_graph_instance_without_stations(graph, stations, depots, cities):
    nodes = np.random.choice(list(set(graph.nodes).difference(set(stations))), depots + cities, replace=False)
    return nodes[:depots], nodes[depots:], np.random.choice(stations, 20, replace=False)


def random_vrp_instance(depots, cities, num_depots, num_cities, num_stations):
    if len(depots) < num_depots:
        locs = np.random.choice(cities, num_depots + num_cities + num_stations - len(depots), replace=False)
        d = np.concatenate((np.array(depots), locs[len(depots) - num_depots:]))
    else:
        locs = np.random.choice(cities, num_cities + num_stations, replace=False)
        d = np.random.choice(depots, num_depots, replace=False)
    return d, locs[:num_cities], locs[num_cities: num_cities + num_stations]


def prepare_data(num):
    if not os.path.isfile(path + '/manhattan.pickle'):
        print('Preparing Manhattan Data')
        graph, data = manhattan(), []
        for _ in tqdm(range(num)):
            depots, cities, stations = random_graph_instance(graph, 5, 30, 20)
            distance = compute_distance_on_graph(graph, depots, cities, stations)
            data.append((depots.tolist(), cities.tolist(), stations.tolist(), distance))
        with open(path + '/manhattan.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/cambridge.pickle'):
        print('Preparing Cambridge Data')
        graph, stations = boston()
        data = []
        for _ in tqdm(range(num)):
            depots, cities, stations = random_graph_instance_without_stations(graph, stations, 5, 30)
            distance = compute_distance_on_graph(graph, depots, cities, stations)
            data.append((depots.tolist(), cities.tolist(), stations, distance))
        with open(path + '/cambridge.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/mdvrp.pickle'):
        print('Preparing MDVRP Data')
        graph, d, c = md_vrp()
        data = []
        for _ in tqdm(range(num)):
            depots, cities, stations = random_vrp_instance(d, c, 5, 30, 20)
            distance = {i: {j: graph.edges[i, j]['weight'] for j in np.concatenate((depots, cities, stations))}
                        for i in np.concatenate((depots, cities, stations))}
            data.append((depots.tolist(), cities.tolist(), stations.tolist(), distance))
        with open(path + '/mdvrp.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/random.pickle'):
        print('Preparing Random Data')
        data = []
        for _ in tqdm(range(num)):
            graph, depots, cities, stations = random_instance(5, 30, 20)
            distance = {i: {j: graph.edges[i, j]['weight'] for j in graph.nodes} for i in graph.nodes}
            data.append((depots.tolist(), cities.tolist(), stations.tolist(), distance))
        with open(path + '/random.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/manhattan_large.pickle'):
        print('Preparing Manhattan Large Data')
        graph, data = manhattan(), []
        for _ in tqdm(range(num)):
            depots, cities, stations = random_graph_instance(graph, 10, 100, 20)
            distance = compute_distance_on_graph(graph, depots, cities, stations)
            data.append((depots.tolist(), cities.tolist(), stations.tolist(), distance))
        with open(path + '/manhattan_large.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/cambridge_large.pickle'):
        print('Preparing Cambridge Large Data')
        graph, stations = boston()
        data = []
        for _ in tqdm(range(num)):
            depots, cities, stations = random_graph_instance_without_stations(graph, stations, 10, 100)
            distance = compute_distance_on_graph(graph, depots, cities, stations)
            data.append((depots.tolist(), cities.tolist(), stations, distance))
        with open(path + '/cambridge_large.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/mdvrp_large.pickle'):
        print('Preparing MDVRP Large Data')
        graph, d, c = md_vrp()
        data = []
        for _ in tqdm(range(num)):
            depots, cities, stations = random_vrp_instance(d, c, 10, 100, 20)
            distance = {i: {j: graph.edges[i, j]['weight'] for j in np.concatenate((depots, cities, stations))}
                        for i in np.concatenate((depots, cities, stations))}
            data.append((depots.tolist(), cities.tolist(), stations.tolist(), distance))
        with open(path + '/mdvrp_large.pickle', 'wb') as f:
            pickle.dump(data, f)
    if not os.path.isfile(path + '/random_large.pickle'):
        print('Preparing Random Large Data')
        data = []
        for _ in tqdm(range(num)):
            graph, depots, cities, stations = random_instance(10, 100, 20)
            distance = {i: {j: graph.edges[i, j]['weight'] for j in graph.nodes} for i in graph.nodes}
            data.append((depots.tolist(), cities.tolist(), stations.tolist(), distance))
        with open(path + '/random_large.pickle', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    prepare_data(1000)
