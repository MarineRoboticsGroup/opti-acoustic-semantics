import argparse
import torch
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def show_similarity_interactive(image_path_a: str, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8'):
    """
     finding similarity between a descriptor in one image to the all descriptors in the other image.
     :param image_path_a: path to first image.
     :param load_size: size of the smaller edge of loaded images. If None, does not resize.
     :param layer: layer to extract descriptors from.
     :param facet: facet to extract descriptors from.
     :param bin: if True use a log-binning descriptor.
     :param stride: stride of the model.
     :param model_type: type of model to extract descriptors from.
     :param num_sim_patches: number of most similar patches to plot.
     :return list of descriptors that should not be used in mapping 
    """
    # extract descriptors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size

    # plot
    fig, axes = plt.subplots(1, 2)
    [axi.set_axis_off() for axi in axes.ravel()]
    visible_patches = []
    radius = patch_size // 2
    # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
    axes[0].imshow(image_pil_a)

    # calculate and plot similarity between image1 descriptors
    similarities = chunk_cosine_sim(descs_a, descs_a)
    curr_similarities = similarities[0, 0, 0, 1:]  # similarity to all spatial descriptors, without cls token
    curr_similarities = curr_similarities.reshape(num_patches_a)
    axes[1].imshow(curr_similarities.cpu().numpy(), cmap='jet')
    plt.draw()

    # start interactive loop
    # get input point from user
    fig.suptitle('Select a point on the left image. \n Right click when finished.', fontsize=16)
    plt.draw()
    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    while len(pts) == 1:
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
        new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
        y_descs_coor = int(new_H / load_size_a[0] * y_coor)
        x_descs_coor = int(new_W / load_size_a[1] * x_coor)

        # reset previous marks
        for patch in visible_patches:
            patch.remove()
            visible_patches = []

        # draw chosen point
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
        axes[0].add_patch(patch)
        visible_patches.append(patch)

        # get and draw current similarities
        raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
        raveled_desc_idx_including_cls = raveled_desc_idx + 1
        curr_similarities = similarities[0, 0, raveled_desc_idx_including_cls, 1:]
        curr_similarities = curr_similarities.reshape(num_patches_a)
        axes[1].imshow(curr_similarities.cpu().numpy(), cmap='jet')
        plt.draw()

        # get input point from user
        fig.suptitle('Select a point on the left image', fontsize=16)
        plt.draw()
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    plt.close()
    return descs_a[0, 0, raveled_desc_idx_including_cls, :].cpu().numpy()


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
