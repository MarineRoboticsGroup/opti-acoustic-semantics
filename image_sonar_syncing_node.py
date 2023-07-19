from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
from sensor_msgs.msg import CompressedImage as RosImageCompressed
from sensor_msgs.msg import Image as RosImage
import numpy as np
import argparse
from PIL import Image
from cosegmentation import find_cosegmentation_ros, draw_cosegmentation_binary_masks, draw_cosegmentation
import torch
import io

import message_filters
from std_msgs.msg import Int32, Float32



bridge = CvBridge()
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


def image_sonar_callback(image_msg, sonar_msg):
      # print(data.encoding)
    # try:
    #     cv_image = bridge.imgmsg_to_cv2(data, "32FC1")
    # except CvBridgeError as e:
    #     print(e)
    
    # img = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
    #img2 = np.array(img, dtype=np.float32) 
    
    with torch.no_grad():

        im_pil = Image.open(io.BytesIO(bytearray(msg.data)))
        # computing cosegmentation
        
        ims_pil = [im_pil] # list for future extension to cosegmentation of multiple images
    
        seg_masks, pil_images = find_cosegmentation_ros(ims_pil, args.elbow, args.load_size, args.layer,
                                                    args.facet, args.bin, args.thresh, args.model_type, args.stride,
                                                    args.votes_percentage, args.sample_interval,
                                                    args.remove_outliers, args.outliers_thresh,
                                                    args.low_res_saliency_maps)#, curr_save_dir)

        # saving cosegmentations
        binary_mask_figs = draw_cosegmentation_binary_masks(seg_masks)
        chessboard_bg_figs = draw_cosegmentation(seg_masks, pil_images)



    image_pub = rospy.Publisher("/kelpie/img_segmented", RosImage, queue_size=10)
    
    im = seg_masks[0].convert('RGB') # assuming single image only in list

    msg = RosImage()
    msg.header.stamp = rospy.Time.now()
    msg.height = im.height
    msg.width = im.width
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 3 * im.width
    msg.data = np.array(im).tobytes()

    image_pub.publish(msg)


    # try:
    #     image_pub.publish(bridge.cv2_to_imgmsg(seg_masks, "32FC1"))
    # except CvBridgeError as e:
    #     print(e)




if __name__ == "__main__":
    rospy.init_node('img_segmentation_node')
    image_topic = "/usb_cam/image_raw/compressed"
    sonar_topic = "/kelpie/sonar"

    image_sub = message_filters.Subscriber(image_topic, RosImageCompressed)
    sonar_sub = message_filters.Subscriber(sonar_topic, )

ts = message_filters.ApproximateTimeSynchronizer([image_sub, sonar_sub], 10, 0.1, allow_headerless=True)
ts.registerCallback(image_sonar_callback)
rospy.spin()
    print("STARTED SUBSCRIBER!")

    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor cosegmentations.')
    # parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    # parser.add_argument('--save_dir', type=str, required=True, help='The root save dir for image sets results.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input images. If None maintains'
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
    parser.add_argument('--sample_interval', default=100, type=int, help="sample every ith descriptor for training"
                                                                         "clustering.")
    parser.add_argument('--remove_outliers', default='False', type=str2bool, help="Remove outliers using cls token.")
    parser.add_argument('--outliers_thresh', default=0.7, type=float, help="Threshold for removing outliers.")
    parser.add_argument('--low_res_saliency_maps', default='True', type=str2bool, help="using low resolution saliency "
                                                                                       "maps. Reduces RAM needs.")

    args = parser.parse_args()


    rospy.spin()