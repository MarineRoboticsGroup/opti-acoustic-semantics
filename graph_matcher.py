import torch # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
import numpy as np
from scipy.spatial.distance import pdist, squareform
import functools

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
        
        subgraph_topic = rospy.get_param("~subgraph_topic", "/landmark_features")
        fullgraph_topic = rospy.get_param("~fullgraph_topic", "/landmark_features")
        
        self.subgraph_sub = message_filters.Subscriber(subgraph_topic, ObjectsVector, queue_size=1)
        self.fullgraph_sub = message_filters.Subscriber(fullgraph_topic, ObjectsVector, queue_size=1)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.subgraph_sub, self.fullgraph_sub), 1, 0.025
        )

        self.sync.registerCallback(self.match)

        

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





# ##############################################################################
# # Generate the larger graph
# # --------------------------
# #
# num_nodes2 = 10
# A2 = torch.rand(num_nodes2, num_nodes2)
# A2 = (A2 + A2.t() > 1.) * (A2 + A2.t()) / 2
# torch.diagonal(A2)[:] = 0
# n2 = torch.tensor([num_nodes2])

# ##############################################################################
# # Generate the smaller graph
# # ---------------------------
# #
# num_nodes1 = 5
# G2 = nx.from_numpy_array(A2.numpy())
# pos2 = nx.spring_layout(G2)
# pos2_t = torch.tensor([pos2[_] for _ in range(num_nodes2)])
# selected = [0] # build G1 as a cluster in visualization
# unselected = list(range(1, num_nodes2))
# while len(selected) < num_nodes1:
#     dist = torch.sum(torch.sum(torch.abs(pos2_t[selected].unsqueeze(1) - pos2_t[unselected].unsqueeze(0)), dim=-1), dim=0)
#     select_id = unselected[torch.argmin(dist).item()] # find the closest node from unselected
#     selected.append(select_id)
#     unselected.remove(select_id)
# selected.sort()
# A1 = A2[selected, :][:, selected]
# X_gt = torch.eye(num_nodes2)[selected, :]
# n1 = torch.tensor([num_nodes1])

# ##############################################################################
# # Visualize the graphs
# # ---------------------
# #
# G1 = nx.from_numpy_array(A1.numpy())
# pos1 = {_: pos2[selected[_]] for _ in range(num_nodes1)}
# color1 = ['#FF5733' for _ in range(num_nodes1)]
# color2 = ['#FF5733' if _ in selected else '#1f78b4' for _ in range(num_nodes2)]
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title('Subgraph 1')
# plt.gca().margins(0.4)
# nx.draw_networkx(G1, pos=pos1, node_color=color1)
# plt.subplot(1, 2, 2)
# plt.title('Graph 2')
# nx.draw_networkx(G2, pos=pos2, node_color=color2)
 
# ##############################################################################
# # We then show how to automatically discover the matching by graph matching.
# #
# # Build affinity matrix
# # ----------------------
# # To match the larger graph and the smaller graph, we follow the formulation of Quadratic Assignment Problem (QAP):
# #
# # .. math::
# #
# #     &\max_{\mathbf{X}} \ \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})\\
# #     s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}
# #
# # where the first step is to build the affinity matrix (:math:`\mathbf{K}`)
# #
# conn1, edge1 = pygm.utils.dense_to_sparse(A1)
# conn2, edge2 = pygm.utils.dense_to_sparse(A2)
# import functools
# gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001) # set affinity function
# K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

# ##############################################################################
# # Visualization of the affinity matrix. For graph matching problem with :math:`N_1` and :math:`N_2` nodes,
# # the affinity matrix has :math:`N_1N_2\times N_1N_2` elements because there are :math:`N_1^2` and
# # :math:`N_2^2` edges in each graph, respectively.
# #
# # .. note::
# #     The diagonal elements of the affinity matrix is empty because there is no node features in this example.
# #
# plt.figure(figsize=(4, 4))
# plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
# plt.imshow(K.numpy(), cmap='Blues')

# ##############################################################################
# # Solve graph matching problem by RRWM solver
# # -------------------------------------------
# # See :func:`~pygmtools.classic_solvers.rrwm` for the API reference.
# #
# X = pygm.rrwm(K, n1, n2)

# ##############################################################################
# # The output of RRWM is a soft matching matrix. Visualization:
# #
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title('RRWM Soft Matching Matrix')
# plt.imshow(X.numpy(), cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')

# ##############################################################################
# # Get the discrete matching matrix
# # ---------------------------------
# # Hungarian algorithm is then adopted to reach a discrete matching matrix
# #
# X = pygm.hungarian(X)

# ##############################################################################
# # Visualization of the discrete matching matrix:
# #
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X.numpy(), cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')

# #############################################################################
# # Match the subgraph
# # -------------------
# # Draw the matching:
# #
# plt.figure(figsize=(8, 4))
# plt.suptitle(f'RRWM Matching Result (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# ax1 = plt.subplot(1, 2, 1)
# plt.title('Subgraph 1')
# plt.gca().margins(0.4)
# nx.draw_networkx(G1, pos=pos1, node_color=color1)
# ax2 = plt.subplot(1, 2, 2)
# plt.title('Graph 2')
# nx.draw_networkx(G2, pos=pos2, node_color=color2)
# for i in range(num_nodes1):
#     j = torch.argmax(X[i]).item()
#     con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
#                           axesA=ax1, axesB=ax2, color="green" if X_gt[i,j] == 1 else "red")
#     plt.gca().add_artist(con)

if __name__ == "__main__":
    rospy.init_node("graph_matcher")
    graph_matcher = GraphMatcher()
    rospy.spin()