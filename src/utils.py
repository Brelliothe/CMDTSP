import math
import networkx as nx
import numpy as np


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def haversine(lon1, lat1, lon2, lat2):
    # Radius of the Earth in kilometers (mean radius)
    radius = 6371.0
    # Convert latitude and longitude from degrees to radians
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)

    x = (lon2_rad - lon1_rad) * math.cos(0.5 * (lat2_rad + lat1_rad))
    y = lat2_rad - lat1_rad

    # Distance in kilometers
    return radius * math.sqrt(x ** 2 + y ** 2)


def compute_distance_on_graph(graph, depots, cities, stations):
    distance = {}
    for i in np.concatenate((depots, cities, stations)):
        distance[i.item()] = {}
        for j in np.concatenate((depots, cities, stations)):
            distance[i.item()][j.item()] = nx.shortest_path_length(graph, i, j, weight='weight')
    return distance
