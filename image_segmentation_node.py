#!/home/singhk/miniconda3/envs/dino-vit-feats-env/bin python

from cv_bridge import CvBridge, CvBridgeError
import cv2

import rospy
from sensor_msgs.msg import CompressedImage as RosImageCompressed
from sensor_msgs.msg import Image as RosImage
from semanticslam_ros.msg import ObjectVector, ObjectsVector
import tf2_ros
#import tf2_geometry_msgs

import message_filters
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import CameraInfo

import numpy as np
import argparse
from PIL import Image
from cosegmentation import find_cosegmentation_ros, draw_cosegmentation_binary_masks, draw_cosegmentation
import torch
import io
from matplotlib import cm
from extractor import ViTExtractor

import pyrealsense2 as rs


# some util functions, probably should move into a utils file
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Thanks Tim! 
def unproject(
    u,
    v,
    z,
    K: np.ndarray,
):
    """
    Given pixel coordinates (u, v) and depth z, return 3D point in camera frame
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    # print(u,v)
    # print(fx, fy, cx, cy)
    print(x, y, z)
    return np.array([x, y, z])

def image_depth_callback(image_msg, depth_msg):
    # global extractor, saliency_extractor, args, bridge, CAM_FOV, CAM_TO_SONAR_TF, SONAR_TO_CAM_TF
      # print(data.encoding)
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, "passthrough")
    except CvBridgeError as e:
        print(e)
    
    im_pil = Image.fromarray(cv_image)
    # print(im_pil.size)
    
    depth_image = bridge.imgmsg_to_cv2(depth_msg, "passthrough")
    depth_array = np.array(depth_image, dtype=np.float32)

    with torch.no_grad():
        
        # convert ROS Compressed image to PIL image
        # im_pil = Image.open(io.BytesIO(bytearray(image_msg.data)))
        # computing cosegmentation
        ims_pil = [im_pil] # list for future extension to cosegmentation of multiple images
    
        # pos centroids are as a fraction of the image size from top left
        seg_masks, pil_images, centroids, pos_centroids, clustered_arrays = find_cosegmentation_ros(extractor, saliency_extractor, ims_pil, args.elbow, args.load_size, args.layer,
                                                    args.facet, args.bin, args.thresh, args.model_type, args.stride,
                                                    args.votes_percentage, args.sample_interval,
                                                    args.remove_outliers, args.outliers_thresh,
                                                    args.low_res_saliency_maps)
        # saving cosegmentations
        binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
        chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)

    
    # get angles, depth for each cluster centroid

    # 0,0 top left, 1,1 bottom right
    # x down, y right, z forwards
    centroids_msg = ObjectsVector()
    centroids_msg.header = image_msg.header

    for i, pos_cent in enumerate(pos_centroids):
        # bearing = pos_cent[0] * CAM_FOV_HOR
        # elevation = pos_cent[1] * CAM_FOV_VERT
        x_pix = int(pos_cent[0] * image_msg.width)
        y_pix = int(pos_cent[1] * image_msg.height) 
        #x_pix = int(pos_cent[0])
        #y_pix = int(pos_cent[1])
        #print(x_pix, y_pix)
        range = depth_array[y_pix, x_pix]/1000.0 # convert to meters
        #print(range)
        x, y, z = unproject(x_pix, y_pix, range, K)
        #print(x, y, z)
        centroid_msg = ObjectVector()
        centroid_msg.geometric_centroid.x = x
        centroid_msg.geometric_centroid.y = y
        centroid_msg.geometric_centroid.z = z
        centroid_msg.latent_centroid = list(centroids[i])
        
        centroids_msg.objects.append(centroid_msg)
        
    # publish 3D positions of cluster centroids

    cluster_pub.publish(centroids_msg)
    

    
    # for publishing segmentation masks (fg/bg)
    im_mask = seg_masks[0].convert('RGB') # assuming single image only in list
    msg_mask = RosImage()
    msg_mask.header.stamp = image_msg.header.stamp
    msg_mask.height = im_mask.height
    msg_mask.width = im_mask.width
    msg_mask.encoding = "rgb8"
    msg_mask.is_bigendian = False
    msg_mask.step = 3 * im_mask.width
    msg_mask.data = np.array(im_mask).tobytes()

    fg_bg_img_pub.publish(msg_mask)    
    
    # for publishing cluster images
    normalizedClusters = (clustered_arrays-np.min(clustered_arrays))/(np.max(clustered_arrays)-np.min(clustered_arrays))
    im = Image.fromarray(np.uint8(cm.jet(normalizedClusters)*255))
    im = im.convert('RGB')
    #im.show()

    msg = RosImage()
    msg.header.stamp = image_msg.header.stamp
    msg.height = im.height
    msg.width = im.width
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 3 * im.width
    msg.data = np.array(im).tobytes()

    cluster_img_pub.publish(msg)


    # try:
    #     image_pub.publish(bridge.cv2_to_imgmsg(seg_masks, "32FC1"))
    # except CvBridgeError as e:
    #     print(e)




if __name__ == "__main__":
    rospy.init_node('img_segmentation_node')

    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor cosegmentations.')
    # parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    # parser.add_argument('--save_dir', type=str, required=True, help='The root save dir for image sets results.')
    parser.add_argument('--load_size', default=350, type=int, help='load size of the input images. If None maintains'
                                                                    'original image size, if int resizes each image'
                                                                    'such that the smaller side is this number.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--thresh', default=0.11, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--elbow', default=0.975, type=float, help='Elbow coefficient for setting number of clusters.')
    parser.add_argument('--votes_percentage', default=85, type=int, help="percentage of votes needed for a cluster to "
                                                                         "be considered salient.")
    parser.add_argument('--sample_interval', default=10, type=int, help="sample every ith descriptor for training"
                                                                         "clustering.")
    parser.add_argument('--remove_outliers', default='False', type=str2bool, help="Remove outliers using cls token.")
    parser.add_argument('--outliers_thresh', default=0.7, type=float, help="Threshold for removing outliers.")
    parser.add_argument('--low_res_saliency_maps', default='True', type=str2bool, help="using low resolution saliency maps. Reduces RAM needs.")
    parser.add_argument('--cam_fov_hor', default=69.4, type=int, help="Camera horizontal field of view in degrees.")
    parser.add_argument('--cam_fov_vert', default=42.5, type=int, help="Camera horizontal field of view in degrees.")


    args = parser.parse_args()
    
    CAM_FOV_HOR = args.cam_fov_hor
    CAM_FOV_VERT = args.cam_fov_vert
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    if args.low_res_saliency_maps:
        saliency_extractor = ViTExtractor(args.model_type, stride=8, device=device)
    else:
        saliency_extractor = extractor
    bridge = CvBridge()
    
    cam_info_topic = "/camera/rgb/camera_info"
    cam_info_msg = rospy.wait_for_message(cam_info_topic, CameraInfo)
    K = np.array(cam_info_msg.K).reshape(3,3)
    print("Camera intrinsics: ", K)

    cluster_img_pub = rospy.Publisher("/camera/img_segmented", RosImage, queue_size=10)
    fg_bg_img_pub = rospy.Publisher("/camera/img_fg_bg", RosImage, queue_size=10)
    cluster_pub = rospy.Publisher("/camera/objects", ObjectsVector, queue_size=10)

    
    image_topic = "/camera/rgb/image_color"
    depth_topic = "/camera/depth/image"

    image_sub = message_filters.Subscriber(image_topic, RosImage)
    depth_sub = message_filters.Subscriber(depth_topic, RosImage)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.1, allow_headerless=True)
    ts.registerCallback(image_depth_callback)
    
    
    
    print("STARTED IMAGE SEGMENTATION NODE!")


    rospy.spin()