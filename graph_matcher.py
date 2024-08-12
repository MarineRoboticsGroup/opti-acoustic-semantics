import torch # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
import numpy as np
from scipy.spatial.distance import pdist, squareform
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
        with open("kitti_map.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.fullgraph = ObjectsVector()
            self.fullgraph.deserialize(bytes)
            self.subgraph = self.fullgraph
        with open("kitti_seq05_colors.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.lm_colors = Marker()
            self.lm_colors.deserialize(bytes)
            
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
    def create_adjacency_matrix(self, points):
        # Calculate pairwise Euclidean distances
        dist_matrix = squareform(pdist(points, 'euclidean'))
        # Normalize distances and create adjacency matrix (optional: apply a threshold for connections)
        adjacency_matrix = torch.tensor(dist_matrix)
        return adjacency_matrix
  
    def match(self, subgraph: ObjectsVector, fullgraph: ObjectsVector) -> None:
        """
        Run graph matching and publish result to /matching topic
        """
        # Extract nodes and edges from subgraph and fullgraph
        
        # Latent centroids are to be used as node features
        subgraph_nodes = [obj.latent_centroid for obj in subgraph.objects][15:20]
        fullgraph_nodes = [obj.latent_centroid for obj in fullgraph.objects]
        subgraph_nodes = torch.tensor(subgraph_nodes)
        fullgraph_nodes = torch.tensor(fullgraph_nodes)
        
        # Geometric centroids are to be used as node positions, which are used to calculate edge features as Euclidean distances
        subgraph_points = [obj.geometric_centroid for obj in subgraph.objects]
        fullgraph_points = [obj.geometric_centroid for obj in fullgraph.objects]

        subgraph_points = np.array([[point.x, point.y, point.z] for point in subgraph_points])[15:20]
        fullgraph_points = np.array([[point.x, point.y, point.z] for point in fullgraph_points])
        
        colors = self.lm_colors.colors
        colors = [(color.r, color.g, color.b, color.a) for color in colors]

        # Create adjacency matrices
        A1 = self.create_adjacency_matrix(subgraph_points)
        A2 = self.create_adjacency_matrix(fullgraph_points)

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
        
        selected = [15, 16, 17, 18, 19]
        X_gt = torch.eye(num_nodes2)[selected, :]

        # full 
        G2 = nx.from_numpy_array(A2.numpy())
        pos2 = nx.spring_layout(G2)
        color1 = ['#FF5733' for _ in range(num_nodes1)]
        color2 = ['#FF5733' if _ in selected else '#1f78b4' for _ in range(num_nodes2)]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Subgraph 1')
        plt.gca().margins(0.4)
        nx.draw_networkx(G1, pos=pos1, node_color=color1)
        plt.subplot(1, 2, 2)
        plt.title('Graph 2')
        nx.draw_networkx(G2, pos=pos2, node_color=color2)
                
        
        # Define the affinity function
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001)
        
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