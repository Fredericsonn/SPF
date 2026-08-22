import heapq
import math
import random
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Basic graph helpers
# -----------------------------

def euclidean(coords, a, b):
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def reverse_graph(graph):
    reversed_g = {node: {} for node in graph}

    for u, neighbors in graph.items():
        for v, w in neighbors.items():
            reversed_g[v][u] = w

    return reversed_g


def dijkstra_all_distances(graph, start):
    distances = {node: float("inf") for node in graph}
    distances[start] = 0

    queue = [(0, start)]

    while queue:
        current_dist, current = heapq.heappop(queue)

        if current_dist > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))

    return distances


# -----------------------------
# Feature extraction
# -----------------------------

def node_degree(graph, node):
    return len(graph[node])


def avg_edge_weight(graph, node):
    weights = list(graph[node].values())

    if not weights:
        return 0

    return sum(weights) / len(weights)


def extract_features(graph, coords, node, target):
    x1, y1 = coords[node]
    x2, y2 = coords[target]

    return [
        euclidean(coords, node, target),
        abs(x1 - x2),
        abs(y1 - y2),
        node_degree(graph, node),
        node_degree(graph, target),
        avg_edge_weight(graph, node),
        avg_edge_weight(graph, target),
    ]


# -----------------------------
# Training data generation
# -----------------------------

def generate_training_data(graph, coords, targets):
    reversed_g = reverse_graph(graph)

    X = []
    y = []

    for target in targets:
        # One reverse Dijkstra gives distance from every node TO this target.
        true_distances_to_target = dijkstra_all_distances(reversed_g, target)

        for node in graph:
            true_distance = true_distances_to_target[node]

            if true_distance == float("inf"):
                continue

            X.append(extract_features(graph, coords, node, target))
            y.append(true_distance)

    return X, y


# -----------------------------
# Learned heuristic
# -----------------------------

class LearnedHeuristic:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, graph, coords, node, target):
        features = extract_features(graph, coords, node, target)
        prediction = self.model.predict([features])[0]

        # Heuristic values should not be negative.
        return max(0, prediction)


# -----------------------------
# A* using learned heuristic
# -----------------------------

def a_star_with_learned_heuristic(graph, coords, start, target, learned_h):
    queue = []

    heapq.heappush(queue, (0, start))

    came_from = {}
    g_score = {node: float("inf") for node in graph}
    g_score[start] = 0

    expanded_nodes = 0

    while queue:
        _, current = heapq.heappop(queue)
        expanded_nodes += 1

        if current == target:
            return g_score[current], expanded_nodes

        for neighbor, weight in graph[current].items():
            tentative_g = g_score[current] + weight

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                h = learned_h.predict(graph, coords, neighbor, target)
                priority = tentative_g + h

                heapq.heappush(queue, (priority, neighbor))

    return float("inf"), expanded_nodes


# -----------------------------
# Example usage
# -----------------------------

graph = {
    1: {2: 4, 3: 2},
    2: {4: 5},
    3: {2: 1, 4: 8, 5: 10},
    4: {6: 3},
    5: {6: 2},
    6: {}
}

coords = {
    1: (0, 0),
    2: (2, 1),
    3: (1, 2),
    4: (4, 2),
    5: (3, 5),
    6: (6, 4),
}

training_targets = [4, 5, 6]

X_train, y_train = generate_training_data(
    graph,
    coords,
    training_targets
)

learned_h = LearnedHeuristic()
learned_h.train(X_train, y_train)

cost, expanded = a_star_with_learned_heuristic(
    graph,
    coords,
    start=1,
    target=6,
    learned_h=learned_h
)

print("ML-A* cost:", cost)
print("Expanded nodes:", expanded)