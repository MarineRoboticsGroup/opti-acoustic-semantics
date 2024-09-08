#!/home/singhk/miniconda3/envs/url/bin python

from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import CompressedImage as RosImageCompressed
from sensor_msgs.msg import Image as RosImage
import tf2_ros

import message_filters
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import CameraInfo
import ros_numpy as rnp

import numpy as np
import argparse
from PIL import Image
import torch
import torchvision
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
import torchvision.transforms.functional as F
import io
from matplotlib import cm



def img_cb(image_msg):
    print("new object message")
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, "passthrough")
    except CvBridgeError as e:
        print(e)
    
    im_pil = Image.fromarray(cv_image).resize((224, 224))
    im_torch = F.to_tensor(im_pil).unsqueeze(0)

    with torch.no_grad():
        class_logits, uncertainties, embeddings = model(im_torch) 
        print(uncertainties)
        
if __name__ == '__main__':
    rospy.init_node('unc_node', anonymous=True)
    bridge = CvBridge()
    model_name = 'vit_small_patch16_224.augreg_in21k'
    model = create_model(model_name, pretrained=True, checkpoint_path='/home/singhk/url/weights/vit_small_checkpoint.pth.tar', unc_depth=2, unc_module='pred-net', unc_width=512)
    model.eval()
    rospy.Subscriber("/usb_cam/image_raw_repub", RosImage, img_cb)
    rospy.spin()