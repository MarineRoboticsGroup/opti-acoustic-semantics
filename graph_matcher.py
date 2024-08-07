import torch # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
import numpy as np
from scipy.spatial.distance import pdist, squareform
import functools
import BytesIO

import rospy
import message_filters
from semanticslam_ros.msg import ObjectsVector, ObjectVector


pygm.set_backend('pytorch') # set default backend for pygmtools
_ = torch.manual_seed(1) # fix random seed

    
class GraphMatcher:
    """
    Matches subgraph to a larger graph using Quadratic Assignment Problem (QAP)
    """
    
    def __init__(self) -> None:
        assert torch.cuda.is_available()
        with open("kitti_map.txt", 'rb') as binary_file:
            serialized_bytes = binary_file.read()
            fullgraph = ObjectsVector()
            fullgraph.deserialize(serialized_bytes)
            print(fullgraph)
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
        subgraph_nodes = [obj.latent_centroid for obj in subgraph.objects]
        fullgraph_nodes = [obj.latent_centroid for obj in fullgraph.objects]
        subgraph_nodes = torch.tensor([subgraph_nodes])
        fullgraph_nodes = torch.tensor([fullgraph_nodes])
        
        # Geometric centroids are to be used as node positions, which are used to calculate edge features as Eucledian distances
        subgraph_points = [obj.geometric_centroid for obj in subgraph.objects]
        fullgraph_points = [obj.geometric_centroid for obj in fullgraph.objects]

        subgraph_points = np.array([[point.x, point.y, point.z] for point in subgraph_points])
        fullgraph_points = np.array([[point.x, point.y, point.z] for point in fullgraph_points])

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

        # Define the affinity function
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001)
        
        # Build affinity matrix
        K = pygm.utils.build_aff_mat(
            node_feat1=subgraph_nodes, 
            edge_feat1=edge1, 
            connectivity1=conn1,
            node_feat2=fullgraph_nodes, 
            edge_feat2=edge2, 
            connectivity2=conn2
        )

        print("A1:\n", A1)
        print("A2:\n", A2)
        print("Connectivity 1:\n", conn1)
        print("Edge 1:\n", edge1)
        print("Connectivity 2:\n", conn2)
        print("Edge 2:\n", edge2)
        
        print("Affinity Matrix:\n", K)


        X = pygm.rrwm(K, n1, n2)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('RRWM Soft Matching Matrix')
        plt.imshow(X.numpy(), cmap='Blues')
        plt.show()


        X = pygm.hungarian(X)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
        plt.imshow(X.numpy(), cmap='Blues')
        plt.subplot(1, 2, 2)
        plt.title('Ground Truth Matching Matrix')
        plt.imshow(X_gt.numpy(), cmap='Blues')


if __name__ == "__main__":
    rospy.init_node("graph_matcher")
    graph_matcher = GraphMatcher()
    rospy.spin()