import torch # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

import functools
from io import BytesIO

import rospy
import message_filters
from semanticslam_ros.msg import ObjectsVector, ObjectVector
from visualization_msgs.msg import Marker

pygm.set_backend('pytorch') # set default backend for pygmtools
_ = torch.manual_seed(1) # fix random seed

    
class GraphMatcher:
    """
    Matches subgraph to a larger graph using Quadratic Assignment Problem (QAP)
    """
    
    def __init__(self) -> None:
        assert torch.cuda.is_available()
        with open("zpool_2obj1loop_part1.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.fullgraph = ObjectsVector()
            self.fullgraph.deserialize(bytes)
            # self.subgraph = self.fullgraph
        with open("zpool_2obj1loop_part2.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.subgraph = ObjectsVector()
            self.subgraph.deserialize(bytes)
        with open("zpool_2obj1loop_part1_colors.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.fullgraph_lm_colors = Marker()
            self.fullgraph_lm_colors.deserialize(bytes)
        with open("zpool_2obj1loop_part2_colors.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.subgraph_lm_colors = Marker()
            self.subgraph_lm_colors.deserialize(bytes)           
            
    # def create_edge_features(node_positions):
    #     """
    #     Create edge features as Euclidean distances between node positions.
        
    #     :param node_positions: array of shape (n, 3) where n is the number of nodes and 3 represents the x, y, z coordinates.
    #     :return: 
    #         - edge_features: array of shape (ne, 1) where ne is the number of edges and 1 represents the Euclidean distance.
    #         - connectivity: array of shape (ne, 2) where ne is the number of edges and 2 represents the indices of connected nodes.
    #     """
    #     # Calculate pairwise Euclidean distances
    #     pairwise_distances = squareform(pdist(node_positions, metric='euclidean'))
        
    #     # Extract upper triangle indices for edge features (to avoid duplicate edges in an undirected graph)
    #     triu_indices = np.triu_indices_from(pairwise_distances, k=1)
        
    #     # Create edge features
    #     edge_features = pairwise_distances[triu_indices].reshape(-1, 1)
        
    #     # Create connectivity information
    #     connectivity = np.vstack(triu_indices).T
        
    #     return edge_features, connectivity

        # Function to create adjacency matrix from geometric centroids
    # def create_adjacency_matrix(self, points, threshold):
    #     # Calculate pairwise Euclidean distances
    #     print(points.shape)
    #     dist_matrix = squareform(pdist(points, 'euclidean'))
    #     dist_matrix[dist_matrix > threshold] = 0
    #     print(dist_matrix.shape)
    #     # Apply threshold: connections above the threshold are set to 0
    #     adjacency_matrix = torch.tensor(dist_matrix)
    #     print(adjacency_matrix.shape)
    #     print(adjacency_matrix)
    #     return adjacency_matrix
    def create_adjacency_matrix(self, points, threshold):
        # Number of points
        n = points.shape[0]

        # Initialize the adjacency matrix with np.inf (representing no connection)
        adjacency_matrix = np.full((n, n), 0)

        # Iterate over all pairs of points
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Euclidean distance between point i and point j
                distance = np.linalg.norm(points[i] - points[j])
                
                # If the distance is less than or equal to the threshold, store the distance
                if distance <= threshold:
                    adjacency_matrix[i, j] = distance
                    adjacency_matrix[j, i] = distance
        
        # Optional: set diagonal to 0 to indicate no self-loop connections
        np.fill_diagonal(adjacency_matrix, 0)

        # # Check if the graph is fully connected
        # num_components, labels = connected_components(adjacency_matrix, directed=False, return_labels=True)

        # # If not fully connected, add the smallest missing edges to make it fully connected
        # while num_components > 1:
        #     for i in range(n):
        #         for j in range(i + 1, n):
        #             if adjacency_matrix[i, j] == 0:
        #                 # Calculate the distance for this unconnected pair
        #                 distance = np.linalg.norm(points[i] - points[j])
        #                 # Add this edge to the graph
        #                 adjacency_matrix[i, j] = distance
        #                 adjacency_matrix[j, i] = distance
        #                 # Recompute the numbefullyof components
        #                 num_components, labels = connected_components(adjacency_matrix, directed=False, return_labels=True)
        #                 if num_components == 1:
        #                     break
        #         if num_components == 1:
        #             break

        return torch.tensor(adjacency_matrix)

    def match(self, subgraph: ObjectsVector, fullgraph: ObjectsVector) -> None:
        """
        Run graph matching and publish result to /matching topic
        """
        
        selected = [11,12,13]
        # Extract nodes and edges from subgraph and fullgraph

        # Latent centroids are to be used as node features
        subgraph_nodes = np.array([obj.latent_centroid for obj in subgraph.objects])[selected]
        fullgraph_nodes = np.array([obj.latent_centroid for obj in fullgraph.objects])
        subgraph_nodes = torch.tensor(subgraph_nodes)
        fullgraph_nodes = torch.tensor(fullgraph_nodes)
        
        # Geometric centroids are to be used as node positions, which are used to calculate edge features as Euclidean distances
        subgraph_points = np.array([obj.geometric_centroid for obj in subgraph.objects])[selected]
        fullgraph_points = np.array([obj.geometric_centroid for obj in fullgraph.objects])

        subgraph_points = np.array([[point.x, point.y, point.z] for point in subgraph_points])
        fullgraph_points = np.array([[point.x, point.y, point.z] for point in fullgraph_points])
        
        colors = self.fullgraph_lm_colors.colors
        colors_fullgraph = np.array([(color.r, color.g, color.b, color.a) for color in colors])
        
        colors = self.subgraph_lm_colors.colors
        colors_subgraph = np.array([(color.r, color.g, color.b, color.a) for color in colors])[selected]    
        
        # Create adjacency matrices
        A1 = self.create_adjacency_matrix(subgraph_points, threshold=1000)
        A2 = self.create_adjacency_matrix(fullgraph_points, threshold=1000)

        # Number of nodes
        num_nodes1 = len(subgraph_nodes)
        num_nodes2 = len(fullgraph_nodes)
        n1 = torch.tensor([num_nodes1])
        n2 = torch.tensor([num_nodes2])

        # Convert dense adjacency matrices to sparse representations
        conn1, edge1 = pygm.utils.dense_to_sparse(A1)
        conn2, edge2 = pygm.utils.dense_to_sparse(A2)

        # Visualize the subgraph and fullgraph
        # sub
        G1 = nx.from_numpy_array(A1.numpy())
        pos1 = nx.spring_layout(G1)
        
        X_gt = torch.eye(num_nodes2)[selected, :]

        # full 
        G2 = nx.from_numpy_array(A2.numpy())
        pos2 = nx.spring_layout(G2)
        # color1 = ['#FF5733' for _ in range(num_nodes1)]
        # color2 = ['#FF5733' if _ in selected else '#1f78b4' for _ in range(num_nodes2)]
        color1 = colors_subgraph
        color2 = colors_fullgraph
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Subgraph 1')
        plt.gca().margins(0.4)
        nx.draw_networkx(G1, pos=pos1, node_color=color1)
        plt.subplot(1, 2, 2)
        plt.title('Graph 2')
        nx.draw_networkx(G2, pos=pos2, node_color=color2)
        plt.show()
        
        # Define the affinity function
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001)
        
        print("Subgraph nodes: ", subgraph_nodes.shape)
        print("Fullgraph nodes: ", fullgraph_nodes.shape)
        print("Edge 1: ", edge1.shape)
        print("Connectivity 1: ", conn1.shape)
        print("Edge 2: ", edge2.shape)
        print("Connectivity 2: ", conn2.shape)
        
        # Build affinity matrix      
        K = pygm.utils.build_aff_mat(subgraph_nodes, edge1, conn1, fullgraph_nodes, edge2, conn2, None, None, None, None, edge_aff_fn=gaussian_aff)
        # K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, None, None, None, None, edge_aff_fn=gaussian_aff)
        plt.figure(figsize=(4, 4))
        plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
        plt.imshow(K.numpy(), cmap='Blues')

        # print("A1:\n", A1)
        # print("A2:\n", A2)
        # print("Connectivity 1:\n", conn1)
        # print("Edge 1:\n", edge1)
        # print("Connectivity 2:\n", conn2)
        # print("Edge 2:\n", edge2)
        print(num_nodes1, num_nodes2)
        print(n1, n2)
        print("Affinity Matrix:\n", K.shape)
        print(n1.dtype, n2.dtype, K.dtype)

        # with torch.set_grad_enabled(False):
        #     X = pygm.ngm(K, n1max=float(num_nodes1), n2max=float(num_nodes2), pretrain='voc')
        #     X = pygm.hungarian(X)
        X = pygm.rrwm(K, n1, n2)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('RRWM Soft Matching Matrix')
        plt.imshow(X.numpy(), cmap='Blues')
        plt.show()


        X = pygm.hungarian(X)
        
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
        # plt.imshow(X.numpy(), cmap='Blues')
        # plt.subplot(1, 2, 2)
        # plt.title('Ground Truth Matching Matrix')
        # plt.imshow(X_gt.numpy(), cmap='Blues')
        
        plt.figure(figsize=(8, 4))
        plt.suptitle(f'RRWM Matching Result')
        ax1 = plt.subplot(1, 2, 1)
        plt.title('Subgraph 1')
        plt.gca().margins(0.4)
        nx.draw_networkx(G1, pos=pos1, node_color=color1)
        ax2 = plt.subplot(1, 2, 2)
        plt.title('Graph 2')
        nx.draw_networkx(G2, pos=pos2, node_color=color2)
        for i in range(num_nodes1):
            j = torch.argmax(X[i]).item()
            con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
                                axesA=ax1, axesB=ax2, color="green" if X_gt[i,j] == 1 else "red")
            plt.gca().add_artist(con)
        plt.show()


if __name__ == "__main__":
    graph_matcher = GraphMatcher()
    graph_matcher.match(graph_matcher.subgraph, graph_matcher.fullgraph)