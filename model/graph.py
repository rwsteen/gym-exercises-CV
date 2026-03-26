import numpy as np
import torch

joints = [
    'head', 
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip', 
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

V = len(joints)

edges = [
    (0, 1),  # head → left_shoulder
    (0, 2),  # head → right_shoulder
    (1, 3),  # left_shoulder → left_elbow
    (3, 5),  # left_elbow → left_wrist
    (2, 4),  # right_shoulder → right_elbow
    (4, 6),  # right_elbow → right_wrist
    (1, 7),  # left_shoulder → left_hip
    (2, 8),  # right_shoulder → right_hip
    (7, 9),  # left_hip → left_knee
    (9, 11), # left_knee → left_ankle
    (8, 10), # right_hip → right_knee
    (10,12), # right_knee → right_ankle
    (7, 8),  # left_hip ↔ right_hip (pelvis)
]

class PennActionGraph:
    def __init__(self, strategy='spatial'):
        self.num_node = 13
        self.edge = edges
        self.strategy = strategy
        # Build adjacency matrix with left hip as center
        self.A = build_penn_action_graph(V=self.num_node, edges=self.edge, strategy=strategy, center=7) 

def build_penn_action_graph(V=V, edges=edges, strategy='spatial', center=7):
    if strategy == 'uniform':
        A = np.zeros((1, V, V))
        for i,j in edges:
            A[0,i,j] = 1
            A[0,j,i] = 1
        return torch.tensor(A, dtype=torch.float32)
    
    elif strategy == 'spatial':
        # compute hop distance from center for each node
        dist = compute_hop_distance(V, edges, center)

        A = np.zeros((3,V,V))

        # split edges into subsets based on distance to center
        for i,j in edges:
            if dist[i] == dist[j]:
                subset = 0
            elif dist[i] < dist[j]:
                subset = 1
            else:
                subset = 2

            A[subset,i,j] = 1
            A[subset,j,i] = 1

        # add self-connections to subset 0
        for i in range(V):
            A[0,i,i] = 1

        return torch.tensor(A, dtype=torch.float32)

def compute_hop_distance(num_nodes, edges, center):
    # Build adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    for i,j in edges:
        A[i,j] = 1
        A[j,i] = 1

    # BFS to compute hop distance from center
    dist = np.full(num_nodes, np.inf)
    dist[center] = 0
    queue = [center]

    while queue:
        node = queue.pop(0)
        for neighbor in range(num_nodes):
            if A[node, neighbor] > 0 and dist[neighbor] == np.inf:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)

    return dist.astype(int)

# if __name__ == "__main__":
#     A = build_penn_action_graph()
#     print(A.shape)  # should be (3, 13, 13)

#     import networkx as nx
#     import matplotlib.pyplot as plt

#     # Your adjacency matrix
#     A = build_penn_action_graph()  # (3, 13, 13)
#     A = A[0].numpy()  # use first subset for visualization

#     # Create a NetworkX graph
#     G = nx.Graph()

#     # Add nodes
#     for i, joint in enumerate(joints):
#         G.add_node(i, label=joint)

#     # Add edges
#     V = len(joints)
#     for i in range(V):
#         for j in range(V):
#             if A[i, j] > 0:
#                 G.add_edge(i, j)

#     # Position nodes manually (optional, to resemble human pose)
#     pos = {
#         0: (0, 5),    # head
#         1: (-2, 4),   # left_shoulder
#         2: (2, 4),    # right_shoulder
#         3: (-3, 3),   # left_elbow
#         4: (3, 3),    # right_elbow
#         5: (-3.5, 2), # left_wrist
#         6: (3.5, 2),  # right_wrist
#         7: (-1.5, 2), # left_hip
#         8: (1.5, 2),  # right_hip
#         9: (-1.5, 1), # left_knee
#         10: (1.5, 1), # right_knee
#         11: (-1.5, 0),# left_ankle
#         12: (1.5, 0)  # right_ankle
#     }

#     # Draw the graph
#     plt.figure(figsize=(8, 6))
#     nx.draw(G, pos, with_labels=True, labels={i: j for i,j in enumerate(joints)}, node_size=800, node_color='skyblue')
#     plt.title("Penn Action Skeleton Graph")
#     plt.show()
#     print("Graph visualization complete.")