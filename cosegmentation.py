import argparse
import torch
from pathlib import Path
from torchvision import transforms
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from typing import List, Tuple
from numpy.linalg import norm

MIN_SIZE = 1000 # in pixels 
sparse_setting = False # if we are in an environment with very few objects, this will be more effective


def find_cosegmentation_ros(extractor: ViTExtractor, saliency_extractor: ViTExtractor, imgs: List[Image.Image], elbow: float = 0.975, load_size: int = 224, layer: int = 11, # TODO extend to multiple images (can we leverage cosegmentation?)
                        facet: str = 'key', bin: bool = False, thresh: float = 0.065, model_type: str = 'dino_vits8',
                        stride: int = 8, votes_percentage: int = 75, sample_interval: int = 100,
                        remove_outliers: bool = False, outliers_thresh: float = 0.7, low_res_saliency_maps: bool = True,
                        save_dir: str = None, removal_obj_codes: List[List[float]] = None, obj_removal_thresh: float = 0.9) -> Tuple[List[Image.Image], List[Image.Image], List[Tuple[int, int]], List[Tuple[float, float]], List[int]]:
    """
    finding cosegmentation of a set of images.
    :param imgs: a list of all the images in Pil format.
    :param elbow: elbow coefficient to set number of clusters.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param votes_percentage: the percentage of positive votes so a cluster will be considered salient.
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param remove_outliers: assume existence of outlier images and remove them from cosegmentation process.
    :param outliers_thresh: threshold on cosine similarity between cls descriptors to determine outliers.
    :param low_res_saliency_maps: Use saliency maps with lower resolution (dramatically reduces GPU RAM needs,
    doesn't deteriorate performance).
    :param save_dir: optional. if not None save intermediate results in this directory.
    :return: a list of segmentation masks, a list of processed pil images, a list of centroids in the feature space, 
    a list of centroids as a fraction of distance across image, the cluster labels (TODO make this work for multiple images)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # extractor = ViTExtractor(model_type, stride, device=device)
    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    # if low_res_saliency_maps:
    #     saliency_extractor = ViTExtractor(model_type, stride=8, device=device)
    # else:
    #     saliency_extractor = extractor
    if remove_outliers:
        cls_descriptors = []
    num_images = len(imgs)
    if save_dir is not None:
        save_dir = Path(save_dir)

    # extract descriptors and saliency maps for each image
    for img in imgs:
        image_batch, image_pil = extractor.preprocess_ros(img, load_size)
        image_pil_list.append(image_pil)
        include_cls = remove_outliers  # removing outlier images requires the cls descriptor.
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls).cpu().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        if remove_outliers:
            cls_descriptor, descs = torch.from_numpy(descs[:, :, 0, :]), descs[:, :, 1:, :]
            cls_descriptors.append(cls_descriptor)
        descriptors_list.append(descs)
        if low_res_saliency_maps:
            if load_size is not None:
                low_res_load_size = (curr_load_size[0] // 2, curr_load_size[1] // 2)
            else:
                low_res_load_size = curr_load_size
            image_batch, _ = saliency_extractor.preprocess_ros(img, low_res_load_size)

        saliency_map = saliency_extractor.extract_saliency_maps(image_batch.to(device)).cpu().numpy()
        curr_sal_num_patches, curr_sal_load_size = saliency_extractor.num_patches, saliency_extractor.load_size
        if low_res_saliency_maps:
            reshape_op = transforms.Resize(curr_num_patches, transforms.InterpolationMode.NEAREST)
            saliency_map = np.array(reshape_op(Image.fromarray(saliency_map.reshape(curr_sal_num_patches)))).flatten()
        else:
            saliency_map = saliency_map[0]
        saliency_maps_list.append(saliency_map)

        # save saliency maps and resized images if needed
        if save_dir is not None:
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(saliency_maps_list[-1].reshape(num_patches_list[-1]), vmin=0, vmax=1, cmap='jet')
            fig.savefig(save_dir / f'{Path(image_path).stem}_saliency_map.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            image_pil.save(save_dir / f'{Path(image_path).stem}_resized.png')

    # cluster all images using k-means:
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
    sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
    all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
    normalized_all_sampled_descriptors = all_sampled_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_sampled_descriptors)  # in-place operation

    sum_of_squared_dists = []
    n_cluster_range = list(range(1, 15))
    for n_clusters in n_cluster_range:
        algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
        algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
        squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
        objective = squared_distances.sum()
        sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
        if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
            break

    centroids = algorithm.centroids
    num_labels = np.max(n_clusters) + 1
    num_descriptors_per_image = [num_patches[0]*num_patches[1] for num_patches in num_patches_list]
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image))


    # use saliency maps to vote for salient clusters
    votes = np.zeros(num_labels)
    for image_labels, saliency_map in zip(labels_per_image, saliency_maps_list):
        for label in range(num_labels):
            label_saliency = saliency_map[image_labels[:, 0] == label].mean()
            if label_saliency > thresh:
                votes[label] += 1
    salient_labels = np.where(votes >= np.ceil(num_images * votes_percentage / 100))
    
    # check if any of the salient labels' latent centroids are in the removal_obj_codes list
    # if so, remove them from the salient labels list
    if removal_obj_codes:
        for obj_code in removal_obj_codes:
            for label in salient_labels:
                # check cosine similarity between centroid and obj_code
                cos_sim = np.dot(obj_code, label)/(norm(obj_code)*norm(label))
                if cos_sim > obj_removal_thresh:
                    salient_labels = np.delete(salient_labels, label)
        
    
    # cluster saliency filtering and visualization
    reshaped_labels = None
    cmap = 'jet' if num_labels > 10 else 'tab10'
    # for img, num_patches, label_per_image in zip(imgs, num_patches_list, labels_per_image):
        #reshaped_labels = label_per_image.reshape(num_patches)
        # normalizedClusters = (reshaped_labels-np.min(reshaped_labels))/(np.max(reshaped_labels)-np.min(reshaped_labels))
        # im = Image.fromarray(np.uint8(cm.jet(normalizedClusters)*255))
        # im = im.convert('RGB')
        # im.show()
        # reshaped_labels += 1 # shift everything up so that 0 is background 
        # saliency_mask = np.isin(label_per_image, salient_labels).reshape(num_patches)
        # salient_reshaped_labels = reshaped_labels * saliency_mask # set non-salient labels to 0
        
        # compute centroids in patch space
        # pos_centroids_sum = np.zeros((num_labels, 3))
        # for x_idx, ys in enumerate(reshaped_labels):
        #     for y_idx, val in enumerate(ys):
        #         pos_centroids_sum[val][0] += x_idx
        #         pos_centroids_sum[val][1] += y_idx
        #         pos_centroids_sum[val][2] += 1  # count for normalization
        # pos_centroids = pos_centroids_sum / pos_centroids_sum[:, 2][:, None]
        # pos_centroids = np.delete(pos_centroids, 0, 0) # remove first row (non-salient clusters). Note this resets the labels to start at 0
        # pos_centroids = np.delete(pos_centroids, -1, 1) # remove count column 

        # pos_centroids[:, 0] /= num_patches[0]
        # pos_centroids[:, 1] /= num_patches[1]
    
    # create masks using the salient labels
    segmentation_masks = []
    pos_centroids = []
    latent_centroids = []
    
    c = 0
    for img, labels, num_patches, load_size in zip(image_pil_list, labels_per_image, num_patches_list, load_size_list):
        c += 1
        reshaped_labels = labels.reshape(num_patches)
        mask = np.isin(labels, salient_labels).reshape(num_patches)
        resized_mask = np.array(Image.fromarray(mask).resize((load_size[1], load_size[0]), resample=Image.LANCZOS))
        
        # instance level segmentations: for all salient labels, run a connected components analysis 
        if not sparse_setting: # if we are in an environment with many (especially overlapping) objects, this will be more effective
            for label in range(num_labels): 
                if label not in salient_labels[0]:
                    pass 
                else:
                    class_mask = np.where(reshaped_labels == label, True, False).reshape(num_patches)
                    resized_class_mask = np.array(Image.fromarray(class_mask).resize((load_size[1], load_size[0]), resample=Image.LANCZOS))

                
                    # run grabcut on the mask (connect up occluded sections etc.)
                    try:
                        # apply grabcut on mask
                        grabcut_kernel_size = (7, 7)
                        kernel = np.ones(grabcut_kernel_size, np.uint8)
                        forground_mask = cv2.erode(np.uint8(resized_class_mask), kernel)
                        forground_mask = np.array(Image.fromarray(forground_mask).resize(img.size, Image.NEAREST))
                        background_mask = cv2.erode(np.uint8(1 - resized_class_mask), kernel)
                        background_mask = np.array(Image.fromarray(background_mask).resize(img.size, Image.NEAREST))
                        full_mask = np.ones((load_size[0], load_size[1]), np.uint8) * cv2.GC_PR_FGD
                        full_mask[background_mask == 1] = cv2.GC_BGD
                        full_mask[forground_mask == 1] = cv2.GC_FGD
                        bgdModel = np.zeros((1, 65), np.float64)
                        fgdModel = np.zeros((1, 65), np.float64)
                        cv2.grabCut(np.array(img), full_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                        grabcut_mask = np.where((full_mask == 2) | (full_mask == 0), 0, 1).astype('uint8')
                    except Exception:
                        # if mask is unfitted from gb (e.g. all zeros) -- don't apply it
                        grabcut_mask = resized_mask.astype('uint8')
                        
                    # run connected components to get instance level segmentations for this label 
                    connectivity = 4
                    cc_num_labels, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(grabcut_mask, connectivity, cv2.CV_32S)
                    for stat, centroid in zip(cc_stats[1:], cc_centroids[1:]): # first stat is background

                        
                        # filters: size, contact with edge of frame 
                        # grabcut shape dimensions are swapped compared to opencv stats 
                        if stat[4] < MIN_SIZE or stat[0] == 0 or stat[1] == 0 or stat[0] + stat[2] == grabcut_mask.shape[1] or stat[1] + stat[3] == grabcut_mask.shape[0]:
                            continue
                        else:
                            # calculate centroid in 3D space, camera frame, percentage of distance across image
                            x_percent = centroid[0]/grabcut_mask.shape[1]
                            y_percent = centroid[1]/grabcut_mask.shape[0]
                            pos_centroids.append([x_percent, y_percent])        
                            
                            # look up  centroid in latent space to obtain latent centroid
                            x_patch = int(centroid[0]/grabcut_mask.shape[1] * num_patches[0])
                            y_patch = int(centroid[1]/grabcut_mask.shape[0] * num_patches[1])
                            
                            # assuming single image only in list
                            print("Centroids shapes info")
                            print(centroids.shape)
                            print(labels.shape)
                            print(labels_per_image[0].shape)
                            print(x_patch, y_patch)
                            latent_centroid = centroids[int(labels_per_image[0][y_patch * num_patches[1] + x_patch])]
                            #latent_centroid = centroids[int(labels_per_image[0][y_patch * x_patch])]
                            latent_centroids.append(latent_centroid)        
        
        # now back to the fg/bg masks 
        try:
            # apply grabcut on mask
            grabcut_kernel_size = (7, 7)
            kernel = np.ones(grabcut_kernel_size, np.uint8)
            forground_mask = cv2.erode(np.uint8(resized_mask), kernel)
            forground_mask = np.array(Image.fromarray(forground_mask).resize(img.size, Image.NEAREST))
            background_mask = cv2.erode(np.uint8(1 - resized_mask), kernel)
            background_mask = np.array(Image.fromarray(background_mask).resize(img.size, Image.NEAREST))
            full_mask = np.ones((load_size[0], load_size[1]), np.uint8) * cv2.GC_PR_FGD
            full_mask[background_mask == 1] = cv2.GC_BGD
            full_mask[forground_mask == 1] = cv2.GC_FGD
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(np.array(img), full_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            grabcut_mask = np.where((full_mask == 2) | (full_mask == 0), 0, 1).astype('uint8')
        except Exception:
            # if mask is unfitted from gb (e.g. all zeros) -- don't apply it
            grabcut_mask = resized_mask.astype('uint8')

        grabcut_mask_img = Image.fromarray(np.array(grabcut_mask, dtype=bool))
        #grabcut_mask_img.show()
        segmentation_masks.append(grabcut_mask_img)
        
        # run connected components on mask to obtain instance level segmentation
        if sparse_setting: # if we are in an environment with very few objects, this will be more effective 
            connectivity = 4
            cc_num_labels, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(grabcut_mask, connectivity, cv2.CV_32S)
            pos_centroids = []
            latent_centroids = []
            for stat, centroid in zip(cc_stats[1:], cc_centroids[1:]): # first stat is background                
                # filters: size, contact with edge of frame 
                # grabcut shape dimensions are swapped compared to opencv stats 
                if stat[4] < MIN_SIZE or stat[0] == 0 or stat[1] == 0 or stat[0] + stat[2] == grabcut_mask.shape[1] or stat[1] + stat[3] == grabcut_mask.shape[0]:
                    continue
                else:
                    # calculate centroid in 3D space, camera frame, percentage of distance across image
                    x_percent = centroid[0]/grabcut_mask.shape[1]
                    y_percent = centroid[1]/grabcut_mask.shape[0]
                    pos_centroids.append([x_percent, y_percent])        
                    
                    # look up  centroid in latent space to obtain latent centroid
                    x_patch = int(centroid[0]/grabcut_mask.shape[1] * num_patches[0])
                    y_patch = int(centroid[1]/grabcut_mask.shape[0] * num_patches[1])
                    
                    # assuming single image only in list
                    latent_centroid = centroids[int(labels_per_image[0][y_patch * x_patch])]
                    latent_centroids.append(latent_centroid)
    
    return segmentation_masks, image_pil_list, latent_centroids, pos_centroids, reshaped_labels 


def draw_cosegmentation(segmentation_masks: List[Image.Image], pil_images: List[Image.Image]) -> List[plt.Figure]:
    """
    Visualizes cosegmentation results on chessboard background.
    :param segmentation_masks: list of binary segmentation masks
    :param pil_images: list of corresponding images.
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    figures = []
    for seg_mask, pil_image in zip(segmentation_masks, pil_images):
        # make bg transparent in image
        np_image = np.array(pil_image)
        np_mask = np.array(seg_mask)
        stacked_mask = np.dstack(3 * [seg_mask])
        masked_image = np.array(pil_image)
        masked_image[~stacked_mask] = 0
        masked_image_transparent = np.concatenate((masked_image, 255. * np_mask.astype(np.int32)[..., None]), axis=-1)

        # create chessboard bg
        chessboard_bg = np.zeros(np_image.shape[:2])
        chessboard_edge = 10
        chessboard_bg[[x // chessboard_edge % 2 == 0 for x in range(chessboard_bg.shape[0])], :] = 1
        chessboard_bg[:, [x // chessboard_edge % 2 == 1 for x in range(chessboard_bg.shape[1])]] = \
            1 - chessboard_bg[:, [x // chessboard_edge % 2 == 1 for x in range(chessboard_bg.shape[1])]]
        chessboard_bg[chessboard_bg == 0] = 0.75
        chessboard_bg = 255. * chessboard_bg

        # show
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(chessboard_bg, cmap='gray', vmin=0, vmax=255)
        ax.imshow(masked_image_transparent.astype(np.int32), vmin=0, vmax=255)
        figures.append(fig)
    return figures


def draw_cosegmentation_binary_masks(segmentation_masks) -> List[plt.Figure]:
    """
    Visualize cosegmentation results as binary masks
    :param segmentation_masks: list of binary segmentation masks
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    figures = []
    for seg_mask in segmentation_masks:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(seg_mask)
        figures.append(fig)
    return figures


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
