#/home/jungseok/venvs/cutler/bin/python

from cv_bridge import CvBridge, CvBridgeError
import cv2

import rospy
from sensor_msgs.msg import Image as RosImage
import tf2_ros
#import tf2_geometry_msgs

import message_filters
from std_msgs.msg import Int32, Float32
from sonar_oculus.msg import OculusPing
from semanticslam_ros.msg import ObjectVector, ObjectsVector
from sensor_msgs.msg import CameraInfo
import object_selection_gui

import numpy as np
import argparse
from PIL import Image
from cosegmentation_maskcut import find_cosegmentation_ros, draw_cosegmentation_binary_masks, draw_cosegmentation
import torch
import io
import os
import sys
from matplotlib import cm
from extractor import ViTExtractor
import yaml

from CutLER.maskcut.demo_maskcut import maskcut_demo
from CutLER.maskcut import dino



# CAM_FOV = 80 # degrees, set in main
# CAM_TO_SONAR_TF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # set in main
# SONAR_TO_CAM_TF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # set in main
THRESHOLD = 0

to_rad = lambda bearing: bearing * np.pi / 18000 # only use for ping message!!!!


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
    return np.array([x, y, z])

def ping_to_range(msg: OculusPing, angle: float) -> float:
    """
    msg: OculusPing message
    angle: angle in degrees 
    Convert sonar ping to range (take most intense return on beam) at given angle.
    """
    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    
    # pre-process ping
    #ping = self.sonar.deconvolve(img)
    ping = img

    angle = angle * np.pi / 180 # convert to radians
    angular_res = 2.268928027592628 / 512 # radians for oculus m1200d assuming even spacing TODO angles aren't evenly spaced for oculus
    r = np.linspace(0, msg.fire_msg.range,num=msg.num_ranges)
    az = to_rad(np.asarray(msg.bearings, dtype=np.float32))

    # image is num_ranges x num_beams
    for beam in range(0, len(az)):
        if (az[beam] >= angle - angular_res/2) and (az[beam] <= angle + angular_res/2):
            #print(az[beam], angle, angular_res/2)
            idx = np.argmax(ping[:, beam])
            if ping[idx, beam] > THRESHOLD:
                # beam range
                br = idx*msg.range_resolution
                return br
            else:
                return False # TODO: return NaN or something else, integrate with gapslam 
        # idx = np.argmax(ping[:, beam])
        # if ping[idx, beam] > self.threshold:
        #     # beam range
        #     br = idx*msg.range_resolution
        #     # beam azimuth
        #     ba = -az[beam] # TODO: confirm
        #     pt = Point()
        #     pt.x = br*np.cos(ba)
        #     pt.y = br*np.sin(ba) 
        #     pt.z = 0.0
        #     if (br <= self.gate_high) and (br >= self.gate_low):
        #       cloud_msg.points.append(pt)


def image_sonar_callback(image_msg, sonar_msg):
    print("Hit callback!")
    # global extractor, saliency_extractor, args, bridge, CAM_FOV, CAM_TO_SONAR_TF, SONAR_TO_CAM_TF
      # print(data.encoding)
    # try:
    #     cv_image = bridge.imgmsg_to_cvtf2_py.2(data, "32FC1")
    # except CvBridgeError as e:
    #     print(e)
    
    # img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
    #img2 = np.array(img, dtype=np.float32) 
    
    with torch.no_grad():

        # use this for compressed images 
        # im_pil = Image.open(io.BytesIO(bytearray(image_msg.data)))

        # use this for uncompressed images        
        try:
            cv_image = bridge.imgmsg_to_cv2(image_msg, "passthrough")
        except CvBridgeError as e:
            print(e)
        im_pil = Image.fromarray(cv_image)    
        ims_pil = [im_pil] # list for future extension to cosegmentation of multiple images


        seg_masks, img_out, centroids, pos_centroids  = maskcut_demo(None, ims_pil, backbone, args.patch_size, 
                                                                args.tau, args.N, args.fixed_size, args.cpu, output_path=None)


        # saving cosegmentations
        # binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
        # chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)


    # transform clusters into sonar frame
    # for now just assume sonar and camera colocated 
    
    # get sonar range for each cluster  
    # publish 3D positions of cluster centroids 

    # 0,0 top left, 1,1 bottom right
    # x down, y right, z forwards
    centroids_msg = ObjectsVector()
    centroids_msg.header = image_msg.header

    for i, pos_cent in enumerate(pos_centroids):
        bearing = pos_cent[0] * CAM_FOV - CAM_FOV/2
        print(bearing) 
        range = ping_to_range(sonar_msg, bearing)
        print(range)

        if range:
            x_pix = int(pos_cent[0] * image_msg.width)
            y_pix = int(pos_cent[1] * image_msg.height) 

            
            x, y, z = unproject(x_pix, y_pix, range, K)
            print(x, y, z)
            centroid_msg = ObjectVector()
            centroid_msg.geometric_centroid.x = x
            centroid_msg.geometric_centroid.y = y
            centroid_msg.geometric_centroid.z = z
            centroid_msg.latent_centroid = list(centroids[i])
            
            centroids_msg.objects.append(centroid_msg)
        
    # publish 3D positions of cluster centroids

    cluster_pub.publish(centroids_msg)
    
    # for publishing segmentation masks (fg/bg)
    im_mask = Image.fromarray(seg_masks[0]*255).convert('RGB')
    msg_mask = RosImage()
    msg_mask.header.stamp = image_msg.header.stamp
    msg_mask.height = im_mask.height
    msg_mask.width = im_mask.width
    msg_mask.encoding = "rgb8"
    msg_mask.is_bigendian = False
    msg_mask.step = 3 * im_mask.width
    msg_mask.data = np.array(im_mask).tobytes()

    fg_bg_img_pub.publish(msg_mask)    
    
    # for publishing segmented images
    im = img_out
    im = im.convert('RGB')
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
    parser.add_argument('--remove_objects_image_dir', default=None, type=str, required=False, help='The root dir of images that contain'
                                                                                    'to remove. A GUI to select objects'
                                                                                    'will appear.')
    parser.add_argument('--load_size', default=180, type=int, help='load size of the input images. If None maintains'
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
    parser.add_argument('--thresh', default=0.065, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--elbow', default=0.975, type=float, help='Elbow coefficient for setting number of clusters.')
    parser.add_argument('--votes_percentage', default=75, type=int, help="percentage of votes needed for a cluster to "
                                                                         "be considered salient.")
    parser.add_argument('--sample_interval', default=10, type=int, help="sample every ith descriptor for training"
                                                                         "clustering.")
    parser.add_argument('--remove_outliers', default='False', type=str2bool, help="Remove outliers using cls token.")
    parser.add_argument('--outliers_thresh', default=0.7, type=float, help="Threshold for removing outliers.")
    parser.add_argument('--low_res_saliency_maps', default='True', type=str2bool, help="using low resolution saliency "
                                                                                       "maps. Reduces RAM needs.")
    parser.add_argument('--cam_fov', default=80, type=int, help="Camera field of view (horizontal) in degrees.")
    parser.add_argument('--cam_calibration_path', default='/home/jungseok/Downloads/bluerov_1080_cal.yaml', type=str, help="Path to camera calibration yaml file.")
    parser.add_argument('--obj_removal_thresh', default=0.9, type=float, help="Cosine similarity threshold for removing objects from cosegmentation.")
    
    
    ########### this is for MaskCUT
    
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8], help='patch size')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')
    parser.add_argument('--tau', type=float, default=0.15, help='threshold used for producing binary graph')

    # additional arguments
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--output_path', type=str,  default='', help='path to save outputs')

    
    
    
    
    args = parser.parse_args()
    
    ############ this is for MaskCUT
    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print (msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()
    ##############



    CAM_FOV = args.cam_fov
    removal_obj_codes = []
    
    # for each image in the directory, bring up a GUI to select objects to remove
    if args.remove_objects_image_dir:
        for image_path in os.listdir(args.remove_objects_image_dir):
            with torch.no_grad():
                removal_obj_code = object_selection_gui.show_similarity_interactive(args.remove_objects_image_dir + "/" + image_path, args.load_size, args.layer, args.facet, 
                                                             args.bin, args.stride, args.model_type)
                removal_obj_codes.append(removal_obj_code)
                
        
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # extractor = ViTExtractor(args.model_type, args.stride, device=device)
    # if args.low_res_saliency_maps:
    #     saliency_extractor = ViTExtractor(args.model_type, stride=8, device=device)
    # else:
    #     saliency_extractor = extractor
    bridge = CvBridge()
    
    
    with open(args.cam_calibration_path,'r') as stream:
        cam_info = yaml.safe_load(stream)
            
    K = np.array(cam_info['camera_matrix']['data']).reshape(3,3)
    print("Camera intrinsics: ", K)

    cluster_img_pub = rospy.Publisher("/camera/img_segmented", RosImage, queue_size=10)
    fg_bg_img_pub = rospy.Publisher("/camera/img_fg_bg", RosImage, queue_size=10)
    cluster_pub = rospy.Publisher("/camera/objects", ObjectsVector, queue_size=10)

    
    image_topic = "/usb_cam/image_raw_repub"
    
    # currently vertical and horizontal swapped due to IP address issue TODO fix 
    sonar_topic = "/sonar_vertical/oculus_node/ping"

    image_sub = message_filters.Subscriber(image_topic, RosImage)
    sonar_sub = message_filters.Subscriber(sonar_topic, OculusPing)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, sonar_sub], 100, 100, allow_headerless=False)
    ts.registerCallback(image_sonar_callback)
    
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # SONAR_TO_CAM_TF = tfBuffer.lookup_transform('sonar_horizontal', 'usb_cam', rospy.Time(0), rospy.Duration(1))
    # CAM_TO_SONAR_TF = tfBuffer.lookup_transform('usb_cam', 'sonar_horizontal', rospy.Time(0), rospy.Duration(1))
    
    
    print("STARTED SUBSCRIBER!")


    rospy.spin()