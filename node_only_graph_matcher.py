import torch # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
from matplotlib.animation import FuncAnimation

import networkx as nx # for plotting graphs
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

import functools
from io import BytesIO
import time 

import rospy
import message_filters
from semanticslam_ros.msg import ObjectsVector, ObjectVector, ObjectsVectorUncertainty, ObjectVectorUncertainty
from visualization_msgs.msg import Marker

pygm.set_backend('pytorch') # set default backend for pygmtools
_ = torch.manual_seed(1) # fix random seed

  

class GraphMatcher:
    """
    Matches subgraph to a larger graph using Quadratic Assignment Problem (QAP)
    """
    
    def __init__(self) -> None:
        assert torch.cuda.is_available()
        with open("kitti_seq05_uncertainties.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.fullgraph = ObjectsVectorUncertainty()
            self.fullgraph.deserialize(bytes)
            # self.subgraph = self.fullgraph
        with open("kitti_seq05_uncertainties.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.subgraph = ObjectsVectorUncertainty()
            self.subgraph.deserialize(bytes)
        with open("kitti_seq05_colors_uncertainties.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.fullgraph_lm_colors = Marker()
            self.fullgraph_lm_colors.deserialize(bytes)
        with open("kitti_seq05_colors_uncertainties.txt", 'rb') as binary_file:
            buf = BytesIO(binary_file.read())
            bytes = buf.getvalue()
            self.subgraph_lm_colors = Marker()
            self.subgraph_lm_colors.deserialize(bytes)
        self.fullgraph_points = []
        self.subgraph_points = []           
            

    def create_adjacency_matrix(self, points, threshold):
        # Number of points
        n = points.shape[0]

        # Initialize the adjacency matrix with np.inf (representing no connection)
        adjacency_matrix = np.full((n, n), 0)
        np.fill_diagonal(adjacency_matrix, 1)

        return torch.tensor(adjacency_matrix)

    def cos_aff_fn(self, feat1, feat2): # feat1 has shape (1, n_1, f), feat2 has shape (1, n_2, f)
    # cosine similarity between feat1 and feat2
        return torch.nn.functional.cosine_similarity(feat1.unsqueeze(2), feat2.unsqueeze(1), dim=-1)

    def cos_aff_fn_weighted(self, tensor1, tensor2):
        # Extract features and uncertainties
        features1 = tensor1[:, :, :-4]  # Shape: [1, 3, 384]
        sigmas1 = tensor1[:, :, -1]     # Shape: [1, 3]
        points1 = tensor1[:, :, -4:-1]  # Shape: [1, 3, 3]
        features2 = tensor2[:, :, :-4]  # Shape: [1, 14, 384]
        sigmas2 = tensor2[:, :, -1]     # Shape: [1, 14]
        points2 = tensor2[:, :, -4:-1]  # Shape: [1, 14, 3]
        
        # Compute cosine similarity between features1 and features2
        cosine_sim = torch.nn.functional.cosine_similarity(features1.unsqueeze(2), features2.unsqueeze(1), dim=-1)
        
        # Combine uncertainties; reshape to broadcast across the cosine similarity
        uncertainty_weights = (sigmas1.unsqueeze(2) + sigmas2.unsqueeze(1)) / 2  # Shape: [1, 3, 14]

        # Apply weights to cosine similarity based on uncertainties
        weighted_cosine_sim = cosine_sim * (1 / (1 + uncertainty_weights))  # Shape: [1, 3, 14]
        
        # calculate euclidean distance between points1 and points2
        euclidean_dist = torch.nn.functional.pairwise_distance(points1.unsqueeze(2), points2.unsqueeze(1), p=2)
        
        # scale euclidean distance to be in the same range as cosine similarity
        euclidean_dist = 1 - euclidean_dist / torch.max(euclidean_dist)
        
        weighted_cosine_sim_and_euclidean_dist = weighted_cosine_sim + euclidean_dist
        
        return weighted_cosine_sim_and_euclidean_dist


    def uncertainty_aff_fn(self, feat1, feat2): # feat1 has shape (n_1, f), feat2 has shape (n_2, f)
        sigma = 1
        sigmas = torch.tensor(feat1.shape[0]*[sigma])
        affn = -torch.nn.functional.pairwise_distance(feat1.unsqueeze(2), feat2.unsqueeze(1), p=2) / (2 * sigmas.unsqueeze(1).unsqueeze(1)**2)
        print("AFFN")
        print(affn.shape)
        return affn

    def bhattacharyya_distance(self, mu1, sigma1, mu2, sigma2):
        """Compute Bhattacharyya distance between two Gaussian distributions.
        
        Args:
            mu1 (torch.Tensor): Mean vector of first Gaussian.
            sigma1 (torch.Tensor): Sigma (standard deviation) of first Gaussian.
            mu2 (torch.Tensor): Mean vector of second Gaussian.
            sigma2 (torch.Tensor): Sigma (standard deviation) of second Gaussian.

        Returns:
            torch.Tensor: Bhattacharyya distance.
        """
        term1 = 0.125 * torch.sum((mu1 - mu2) ** 2 / (sigma1 ** 2 + sigma2 ** 2), dim=-1)
        term2 = 0.5 * torch.sum(torch.log((sigma1 ** 2 + sigma2 ** 2) / (torch.sqrt(sigma1 ** 2 * sigma2 ** 2))), dim=-1)
        return term1 + term2

    def bhattacharyya_aff_fn(self, tensor1, tensor2):
        """Compute pairwise Bhattacharyya distances between two tensors.

        Args:
            tensor1 (torch.Tensor): Tensor of shape [1, 3, 385].
            tensor2 (torch.Tensor): Tensor of shape [1, 14, 385].

        Returns:
            torch.Tensor: Tensor of pairwise Bhattacharyya distances.
        """
        # Extract feature vectors and sigma values
        features1 = tensor1[:, :, :-1]  # Shape: [1, 3, 384]
        sigmas1 = tensor1[:, :, -1]      # Shape: [1, 3]
        features2 = tensor2[:, :, :-1]  # Shape: [1, 14, 384]
        sigmas2 = tensor2[:, :, -1]      # Shape: [1, 14]

        # Compute pairwise Bhattacharyya distances
        distances = torch.zeros(features1.size(1), features2.size(1))  # Shape: [2, 11]

        for i in range(features1.size(1)):
            mu1 = features1[:, i, :]  # Shape: [1, 384]
            sigma1 = sigmas1[:, i]     # Shape: [1]
            for j in range(features2.size(1)):
                mu2 = features2[:, j, :]  # Shape: [1, 384]
                sigma2 = sigmas2[:, j]     # Shape: [1]
                distances[i, j] = self.bhattacharyya_distance(mu1, sigma1, mu2, sigma2)
        distances = distances.unsqueeze(0)
        return distances

    def mahalanobis_dist_aff_fn(self, tensor1, tensor2): # feat1 has shape (1, n_1, f), feat2 has shape (1, n_2, f)
        # Mahalanobis distance affinity between feat1 and feat2
        # Extract feature vectors and sigma values
        features1 = tensor1[:, :, :-1]  # Shape: [1, 3, 384]
        sigmas1 = tensor1[:, :, -1]      # Shape: [1, 3]
        features2 = tensor2[:, :, :-1]  # Shape: [1, 14, 384]
        sigmas2 = tensor2[:, :, -1]      # Shape: [1, 14]
        
        # computer pairwise Mahalanobis distances
        distances = torch.zeros(features1.size(1), features2.size(1))
        
        for i in range(features1.size(1)):
            mu1 = features1[:, i, :]
            sigma1 = sigmas1[:, i]
            for j in range(features2.size(1)):
                mu2 = features2[:, j, :]
                sigma2 = sigmas2[:, j]
                distances[i, j] = torch.nn.functional.pairwise_distance(mu1, mu2, p=2) / torch.sqrt(sigma1**2 + sigma2**2)
        distances = distances.unsqueeze(0)
        return distances
    
    def match(self, subgraph: ObjectsVectorUncertainty, fullgraph: ObjectsVectorUncertainty) -> None:
        """
        Run graph matching and publish result to /matching topic
        """
        # day 1
        # selected = [7,8,9,10,11]
        
        # 2obj2loop
        # selected = [0,1,2,3,4]
        
        # 2obj1loop
        # selected = [8,9,10,11,12]
        
        # KITTI seq 05
        # location 1: 380-400
        selected = [380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399]
        # location 2: 430-450
        # selected = [430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449]
        # location 3: 540-560
        # selected = [540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559]
        # location 4: 580-600
        # selected = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599]

        # Extract nodes and edges from subgraph and fullgraph

        # Latent centroids are to be used as node features
        # Super hacky way to store uncertainty in the last dimension of the node features since it needs to be a tensor
        fullgraph_nodes = np.array([list(obj.latent_centroid) +  list(np.array([obj.geometric_centroid.x, obj.geometric_centroid.y, obj.geometric_centroid.z])) + [obj.uncertainty] for obj in fullgraph.objects])
        subgraph_nodes = np.array([list(obj.latent_centroid) + list(np.array([obj.geometric_centroid.x, obj.geometric_centroid.y, obj.geometric_centroid.z])) + [obj.uncertainty] for obj in subgraph.objects])[selected]

        # subgraph_nodes = np.array([list(obj.latent_centroid) for obj in subgraph.objects])[selected]
        # fullgraph_nodes = np.array([list(obj.latent_centroid) for obj in fullgraph.objects])

        subgraph_nodes = torch.tensor(subgraph_nodes)
        fullgraph_nodes = torch.tensor(fullgraph_nodes)
        
        # Geometric centroids are to be used as node positions, which are used to calculate edge features as Euclidean distances
        self.subgraph_points = np.array([obj.geometric_centroid for obj in subgraph.objects])[selected]
        self.fullgraph_points = np.array([obj.geometric_centroid for obj in fullgraph.objects])

        self.subgraph_points = np.array([[point.x, point.y, point.z] for point in self.subgraph_points])
        self.fullgraph_points = np.array([[point.x, point.y, point.z] for point in self.fullgraph_points])
        
        colors = self.fullgraph_lm_colors.colors
        colors_fullgraph = np.array([(color.r, color.g, color.b, color.a) for color in colors])
        
        colors = self.subgraph_lm_colors.colors
        colors_subgraph = np.array([(color.r, color.g, color.b, color.a) for color in colors])[selected]    
        
        # Create adjacency matrices
        t0 = time.time()
        A1 = self.create_adjacency_matrix(self.subgraph_points, threshold=1)
        A2 = self.create_adjacency_matrix(self.fullgraph_points, threshold=1)
        t1 = time.time()
        
        print(f"Time taken to create adjacency matrices: {t1 - t0:.6f} s")

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
        
        # For 2obj2loop and KITTI
        X_gt = torch.eye(num_nodes2)[selected, :]

        # for day 1
        # X_gt = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        #                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])[5 - len(selected):]

        # for 2obj1loop
        # X_gt = torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])[5 - len(selected):]


        # for 1obj1loop
        # X_gt = torch.tensor([[0., 0., 0., 0., 1.],
        #                      [1., 0., 0., 0., 0.],
        #                      [0., 0., 0., 1., 0.],
        #                      [0., 0., 1., 0., 0.],
        #                      [0., 1., 0., 0., 0.]])[5 - len(selected):]



        # G2 = nx.from_numpy_array(A2.numpy())
        # pos2 = nx.spring_layout(G2)
        # # color1 = ['#FF5733' for _ in range(num_nodes1)]
        # # color2 = ['#FF5733' if _ in selected else '#1f78b4' for _ in range(num_nodes2)]
        # color1 = colors_subgraph
        # color2 = colors_fullgraph
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title('Subgraph 1')
        # plt.gca().margins(0.4)
        # nx.draw_networkx(G1, pos=pos1, node_color=color1)
        # plt.subplot(1, 2, 2)
        # plt.title('Graph 2')
        # nx.draw_networkx(G2, pos=pos2, node_color=color2)
        # plt.show()
        
        # Define the affinity function
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)
        
        # print("Subgraph nodes: ", subgraph_nodes.shape)
        # print("Fullgraph nodes: ", fullgraph_nodes.shape)
        # print("Edge 1: ", edge1.shape)
        # print("Connectivity 1: ", conn1.shape)
        # print("Edge 2: ", edge2.shape)
        # print("Connectivity 2: ", conn2.shape)
        
        # Build affinity matrix      
        # print(conn1.shape, edge1.shape, conn2.shape, edge2.shape)
        t0 = time.time()
        K = pygm.utils.build_aff_mat(subgraph_nodes, edge1, conn1, fullgraph_nodes, edge2, conn2, None, None, None, None, node_aff_fn=self.cos_aff_fn_weighted, edge_aff_fn=gaussian_aff)
        # K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, None, None, None, None, node_aff_fn=cos_aff_fn_weighted, edge_aff_fn=gaussian_aff)
        # K = pygm.utils.build_aff_mat(subgraph_nodes, None, conn1, fullgraph_nodes, None, conn2, None, None, None, None, node_aff_fn=cos_aff_fn_weighted, edge_aff_fn=gaussian_aff)
        t1 = time.time()
        print(f"Time taken to build affinity matrix: {t1 - t0:.6f} s")

        # plt.figure(figsize=(4, 4))
        # plt.title(f'Affinity Matrix')
        # plt.imshow(K.numpy(), cmap='Blues')

        # print("A1:\n", A1)
        # print("A2:\n", A2)
        # print("Connectivity 1:\n", conn1)
        # print("Edge 1:\n", edge1)
        # print("Connectivity 2:\n", conn2)
        # print("Edge 2:\n", edge2)
        # print(num_nodes1, num_nodes2)
        # print(n1, n2)
        # print("Affinity Matrix:\n", K.shape)
        # print(n1.dtype, n2.dtype, K.dtype)

        # with torch.set_grad_enabled(False):
        #     X = pygm.ngm(K, n1max=float(num_nodes1), n2max=float(num_nodes2), pretrain='voc')
        #     X = pygm.hungarian(X)
        t0 = time.time()
        intermediate_results = pygm.rrwm(K.float(), n1, n2, max_iter=20)
        X = intermediate_results[-1]
        t1 = time.time()
        
        print(f"Time taken: {t1 - t0:.6f} s")
        
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title('RRWM Soft Matching Matrix')
        # plt.imshow(X.numpy(), cmap='Blues')
        # plt.show()

        # Create a figure and axis for the plot
        # fig, ax = plt.subplots()
        # cax = ax.matshow(intermediate_results[0], cmap='Blues')
        # fig.colorbar(cax)

        # def update(frame):
        #     """
        #     Update function for the animation.
        #     """
        #     cax.set_data(intermediate_results[frame])
        #     ax.set_title(f"RRWM Iteration {frame + 1}")
        #     return cax,

        # # Create the animation
        # anim = FuncAnimation(fig, update, frames=len(intermediate_results), blit=False)

        # # To display the animation in a Jupyter Notebook
        # plt.show()

        # # If you want to save the animation, use the following line (uncomment if needed)
        # anim.save('rrwm_animation.mp4', writer='ffmpeg')
                

        X = pygm.hungarian(X)
        print(X)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
        plt.imshow(X.numpy(), cmap='Blues')
        plt.subplot(1, 2, 2)
        plt.title('Ground Truth Matching Matrix')
        plt.imshow(X_gt.numpy(), cmap='Blues')
        plt.show()
        
        # plt.figure(figsize=(8, 4))
        # plt.suptitle(f'RRWM Matching Result')
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
        #                         axesA=ax1, axesB=ax2, color="green" if X_gt[i,j] == 1 else "red")
        #     plt.gca().add_artist(con)
        # plt.show()
            

if __name__ == "__main__":
    graph_matcher = GraphMatcher()
    graph_matcher.match(graph_matcher.subgraph, graph_matcher.fullgraph)