from cv_bridge import CvBridge, CvBridgeError
import cv2

import rospy
from sensor_msgs.msg import CompressedImage as RosImageCompressed
from sensor_msgs.msg import Image as RosImage
import tf2_ros
#import tf2_geometry_msgs

import message_filters
from std_msgs.msg import Int32, Float32
from sonar_oculus.msg import OculusPing

import numpy as np
import argparse
from PIL import Image
from cosegmentation import find_cosegmentation_ros, draw_cosegmentation_binary_masks, draw_cosegmentation
import torch
import io
from matplotlib import cm
from extractor import ViTExtractor




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
            print(az[beam], angle, angular_res/2)
            idx = np.argmax(ping[:, beam])
            if ping[idx, beam] > THRESHOLD:
                # beam range
                br = idx*msg.range_resolution
                return br
            else:
                return br # TODO: return NaN or something else, integrate with gapslam 
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
    # global extractor, saliency_extractor, args, bridge, CAM_FOV, CAM_TO_SONAR_TF, SONAR_TO_CAM_TF
      # print(data.encoding)
    # try:
    #     cv_image = bridge.imgmsg_to_cvtf2_py.2(data, "32FC1")
    # except CvBridgeError as e:
    #     print(e)
    
    # img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
    #img2 = np.array(img, dtype=np.float32) 
    
    with torch.no_grad():

        im_pil = Image.open(io.BytesIO(bytearray(image_msg.data)))
        # computing cosegmentation
        
        ims_pil = [im_pil] # list for future extension to cosegmentation of multiple images
    
        seg_masks, pil_images, centroids, pos_centroids, clustered_arrays = find_cosegmentation_ros(extractor, saliency_extractor, ims_pil, args.elbow, args.load_size, args.layer,
                                                    args.facet, args.bin, args.thresh, args.model_type, args.stride,
                                                    args.votes_percentage, args.sample_interval,
                                                    args.remove_outliers, args.outliers_thresh,
                                                    args.low_res_saliency_maps)#, curr_save_dir)
        
        # saving cosegmentations
        binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
        chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)


    # transform clusters into sonar frame
    # TODO for now just assume sonar and camera colocated 
    
    # get sonar range for each cluster

    for centroid in pos_centroids:
        bearing = centroid[0] * CAM_FOV
        print(bearing) 
        range = ping_to_range(sonar_msg, bearing)
        print(range)
    
    # publish 3D positions of cluster centroids 
    # TODO what does DCSAM want for message type?


    cluster_img_pub = rospy.Publisher("/usb_cam/img_segmented", RosImage, queue_size=10)
    fg_bg_img_pub = rospy.Publisher("/usb_cam/img_fg_bg", RosImage, queue_size=10)
    
    # for publishing segmentation masks (fg/bg)
    im_mask = seg_masks[0].convert('RGB') # assuming single image only in list
    msg_mask = RosImage()
    msg_mask.header.stamp = rospy.Time.now()
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

    msg = RosImage()
    msg.header.stamp = rospy.Time.now()
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
    parser.add_argument('--load_size', default=360, type=int, help='load size of the input images. If None maintains'
                                                                    'original image size, if int resizes each image'
                                                                    'such that the smaller side is this number.')
    parser.add_argument('--stride', default=8, type=int, help="""stride of first convolution layer. 
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
    parser.add_argument('--cam_fov', default=80, type=int, help="Camera field of view in degrees.")

    args = parser.parse_args()
    
    CAM_FOV = args.cam_fov
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    if args.low_res_saliency_maps:
        saliency_extractor = ViTExtractor(args.model_type, stride=8, device=device)
    else:
        saliency_extractor = extractor
    bridge = CvBridge()
    
    image_topic = "/usb_cam/image_raw/compressed"
    sonar_topic = "/sonar_horizontal/oculus_node/ping"

    image_sub = message_filters.Subscriber(image_topic, RosImageCompressed)
    sonar_sub = message_filters.Subscriber(sonar_topic, OculusPing)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, sonar_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(image_sonar_callback)
    
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    SONAR_TO_CAM_TF = tfBuffer.lookup_transform('sonar_horizontal', 'usb_cam', rospy.Time(0), rospy.Duration(1))
    CAM_TO_SONAR_TF = tfBuffer.lookup_transform('usb_cam', 'sonar_horizontal', rospy.Time(0), rospy.Duration(1))
    
    
    print("STARTED SUBSCRIBER!")


    rospy.spin()