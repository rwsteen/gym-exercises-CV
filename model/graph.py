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

def build_penn_action_graph(V=V, edges=edges, strategy='spatial'):
    if strategy == 'uniform':
        A = np.zeros((1, V, V))
        for i,j in edges:
            A[0,i,j] = 1
            A[0,j,i] = 1
        return torch.tensor(A, dtype=torch.float32)
    
    elif strategy == 'spatial':
        # ST-GCN standard: K=3
        A = np.zeros((3,V,V))
        for i,j in edges:
            # put all edges in subset 0 (root/center)
            A[0,i,j] = 1
            A[0,j,i] = 1
        return torch.tensor(A, dtype=torch.float32)

class PennActionGraph:
    def __init__(self, strategy='spatial'):
        self.num_node = 13
        self.edge = edges
        self.strategy = strategy
        self.A = build_penn_action_graph(V=self.num_node, edges=self.edge, strategy=strategy)

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