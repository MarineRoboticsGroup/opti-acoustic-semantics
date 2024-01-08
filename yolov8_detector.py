#!/usr/bin/env python3

import pathlib

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from semanticslam_ros.msg import ObjectsVector, ObjectVector
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO
from PIL import Image as PILImage
import sys

np.set_printoptions(threshold=sys.maxsize)


CONF_THRESH = 0.25  # Confidence threshold used for YOLO, default is 0.25
EMBEDDING_LEN = 512  # Length of the embedding vector, default is 512


def unproject(u, v, depth, cam_info):
    """
    Unproject a single pixel to 3D space
    """
    fx = cam_info.K[0]
    fy = cam_info.K[4]
    cx = cam_info.K[2]
    cy = cam_info.K[5]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    print("x: ", x)
    print("y: ", y)
    print("depth: ", depth)
    print("fx: ", fx)
    print("fy: ", fy)
    print("cx: ", cx)
    print("cy: ", cy)
    return x, y, depth


class ClosedSetDetector:
    """
    This holds an instance of YoloV8 and runs inference
    """

    def __init__(self) -> None:
        assert torch.cuda.is_available()
        model_file = pathlib.Path(__file__).parent / "../../yolo/yolov8m-seg.pt"
        self.model = YOLO(model_file)
        rospy.loginfo("Model loaded")
        self.objs_pub = rospy.Publisher("/camera/objects", ObjectsVector, queue_size=10)
        self.br = CvBridge()

        # Set up synchronized subscriber 
        # REALSENSE PARAMS
        # cam_info_topic = rospy.get_param("cam_info_topic", "/camera/color/camera_info")
        # rgb_topic = rospy.get_param("rgb_topic", "/camera/color/image_raw")
        # depth_topic = rospy.get_param(
        #     "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        # )
        
        # Set up synchronized subscriber
        # TUM PARAMS 
        cam_info_topic = rospy.get_param("cam_info_topic", "/camera/rgb/camera_info")
        rgb_topic = rospy.get_param("rgb_topic", "/camera/rgb/image_color")
        depth_topic = rospy.get_param(
            "depth_topic", "/camera/depth/image"
        )
        self.cam_info_sub = message_filters.Subscriber(
            cam_info_topic, CameraInfo, queue_size=1
        )
        self.rgb_img_sub = message_filters.Subscriber(rgb_topic, Image, queue_size=1)
        self.depth_img_sub = message_filters.Subscriber(
            depth_topic, Image, queue_size=1
        )

        # Synchronizer for RGB and depth images
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.cam_info_sub, self.rgb_img_sub, self.depth_img_sub), 1, 0.025
        )

        self.sync.registerCallback(self.forward_pass)

    def forward_pass(self, cam_info: CameraInfo, rgb: Image, depth: Image) -> None:
        """
        Run YOLOv8 on the image and extract all segmented masks
        """

        objects = ObjectsVector()
        objects.header = rgb.header
        objects.objects = []

        image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        depth_m = (
            self.br.imgmsg_to_cv2(depth, desired_encoding="passthrough") # / 1000.0 # for TUM depth is in meters already, for realsense it is in mm
        )  # Depth in meters
        # depth_m = cv2.resize(depth_m, dsize=(1280, 736), interpolation=cv2.INTER_NEAREST) # do this for realsense (img dim not a multiple of max stride length 32)

        # Run inference args: https://docs.ultralytics.com/modes/predict/#inference-arguments
        # results = self.model(image_cv, verbose=False, conf=CONF_THRESH, imgsz=(736, 1280))[0] do this for realsense (img dim not a multiple of max stride length 32)
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]

        # Extract segmentation masks
        if (results.boxes is None) or (results.masks is None):
            return

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = PILImage.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            # im.save('results.jpg')  # save image

        masks = results.masks.data.cpu().numpy()
        class_ids = results.boxes.cls.data.cpu().numpy()
        bboxes = results.boxes.xyxy.data.cpu().numpy()
        confs = results.boxes.conf.data.cpu().numpy()
        if len(masks) == 0:
            return
        for mask, class_id, bboxes, conf in zip(masks, class_ids, bboxes, confs):
            # ---- Object Vector ----
            object = ObjectVector()
            class_id = int(class_id)
            print(np.shape(mask))
            print(np.shape(depth_m))
            print(class_id)
            print(conf)
            mask = mask > 0  # Convert to binary 
            #print(mask)
            obj_depth = np.nanmean(depth_m[mask], dtype=float)
            obj_centroid = np.mean(np.argwhere(mask), axis=0)
            print(obj_centroid)

            # Unproject centroid to 3D
            x, y, z = unproject(obj_centroid[1], obj_centroid[0], obj_depth, cam_info)
            object.geometric_centroid.x = x
            object.geometric_centroid.y = y
            object.geometric_centroid.z = z
            
            if (conf < .8):
                object.geometric_centroid.x = np.nan
                object.geometric_centroid.y = np.nan
                object.geometric_centroid.z = np.nan

            object.latent_centroid = np.zeros(EMBEDDING_LEN)
            assert class_id < EMBEDDING_LEN, "Class ID > length of vector"
            object.latent_centroid[class_id] = 1

            objects.objects.append(object)
            

        self.objs_pub.publish(objects)


if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    rospy.spin()