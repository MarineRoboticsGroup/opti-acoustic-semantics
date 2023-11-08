#!/usr/bin/env python3

import pathlib

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from osoda.msg import Detection, Frame, ObjectsVector, ObjectVector
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

CONF_THRESH = 0.25  # Confidence threshold used for YOLO, default is 0.25
EMBEDDING_LEN = 384  # Length of the embedding vector, default is 512


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
        self.objs_pub = rospy.Publisher("objects", ObjectsVector, queue_size=10)
        frame_pub_name = rospy.get_param("frame_pub_name", "frames")
        self.frame_pub = rospy.Publisher(frame_pub_name, Frame, queue_size=10)
        self.br = CvBridge()

        # Set up synchronized subscriber
        cam_info_topic = rospy.get_param("cam_info_topic", "/camera/color/camera_info")
        rgb_topic = rospy.get_param("rgb_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
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

        # ---- Frame ---- // Remove if not using for OSODA
        frame = Frame()
        frame.header = rgb.header
        frame.frame_id = rgb.header.seq
        frame.detections = []
        frame.img_height = rgb.height
        frame.img_width = rgb.width
        frame.num_detections = 0
        # ---- Frame ----

        image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        depth_m = (
            self.br.imgmsg_to_cv2(depth, desired_encoding="passthrough") / 1000.0
        )  # Depth in meters

        # Run inference args: https://docs.ultralytics.com/modes/predict/#inference-arguments
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]
        # Extract segmentation masks
        if (results.boxes is None) or (results.masks is None):
            return

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
            mask = mask > 0  # Convert to binary array
            obj_depth = np.mean(depth_m[mask], dtype=float)
            obj_centroid = np.mean(np.argwhere(mask), axis=0)

            # Unproject centroid to 3D
            x, y, z = unproject(obj_centroid[1], obj_centroid[0], obj_depth, cam_info)
            object.geometric_centroid.x = x
            object.geometric_centroid.y = y
            object.geometric_centroid.z = z

            object.latent_centroid = np.zeros(EMBEDDING_LEN)
            assert class_id < EMBEDDING_LEN, "Class ID > length of vector"
            object.latent_centroid[class_id] = 1

            objects.objects.append(object)
            # ---- Object Vector ----

            # ---- Frame ---- // Remove if not using for OSODA
            # Detection
            detection = Detection()
            detection.depth = z
            detection.descriptor = np.zeros(512)
            detection.descriptor[:EMBEDDING_LEN] = object.latent_centroid
            detection.det_score = conf
            detection.x1 = bboxes[0]
            detection.y1 = bboxes[1]
            detection.x2 = bboxes[2]
            detection.y2 = bboxes[3]

            frame.detections.append(detection)
            frame.num_detections = frame.num_detections + 1
            # ---- Frame ----

        self.objs_pub.publish(objects)
        self.frame_pub.publish(frame)  # ---- Frame ---- Remove if not using for OSODA


if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    rospy.spin()