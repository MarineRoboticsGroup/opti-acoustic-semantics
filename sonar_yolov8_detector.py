import pathlib

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from semanticslam_ros.msg import ObjectsVector, ObjectVector
from sensor_msgs.msg import CameraInfo, Image
from sonar_oculus.msg import OculusPing
from ultralytics import YOLO
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys
import yaml

np.set_printoptions(threshold=sys.maxsize)



# CAM_FOV = 80 # degrees, set in main
# CAM_TO_SONAR_TF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # set in main
# SONAR_TO_CAM_TF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # set in main
THRESHOLD = 0

CONF_THRESH = 0.25  # Confidence threshold used for YOLO, default is 0.25
EMBEDDING_LEN = 512  # Length of the embedding vector, default is 512


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
    print(u, cx, z, fx)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

def ping_to_range(msg: OculusPing, angle: float) -> float:
    """
    msg: OculusPing message
    angle: angle in degrees 
    Convert sonar ping to range (take most intense return on beam) at given angle.
    """
    img = bridge.compressed_imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    #img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")

    # pre-process ping
    #ping = self.sonar.deconvolve(img)
    ping = img
    #print(ping)

    angle = angle * np.pi / 180 # convert to radians
    angular_res = 2.268928027592628 / 512 # radians for oculus m1200d lf and m750d hf/lf assuming even spacing
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



class ClosedSetDetector:
    """
    This holds an instance of YoloV8 and runs inference
    """

    def __init__(self) -> None:
        assert torch.cuda.is_available()
        model_file = "/home/singhk/topside_ws/src/maxmixtures/opti-acoustic-semantics/runs/detect/train/weights/last.pt"
        self.model = YOLO(model_file)
        rospy.loginfo("Model loaded")
        self.objs_pub = rospy.Publisher("/camera/objects", ObjectsVector, queue_size=10)
        self.img_pub = rospy.Publisher("/camera/yolo_img", RosImage, queue_size=10)
        self.br = CvBridge()

        # Set up synchronized subscriber 
        # sdr bluerov params 
        rgb_topic = rospy.get_param("rgb_topic", "/usb_cam/image_raw_repub")
        depth_topic = rospy.get_param(
            "depth_topic", "/sonar_vertical/oculus_node/ping"
        )

        self.rgb_img_sub = message_filters.Subscriber(rgb_topic, Image, queue_size=1)
        self.depth_img_sub = message_filters.Subscriber(
            depth_topic, OculusPing, queue_size=1
        )

        # Synchronizer for RGB and depth images
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.rgb_img_sub, self.depth_img_sub), 100, 1
        )

        self.sync.registerCallback(self.forward_pass)

    def forward_pass(self, rgb: Image, depth: OculusPing) -> None:
        """
        Run YOLOv8 on the image and extract all segmented masks
        """

        objects = ObjectsVector()
        objects.header = rgb.header
        objects.objects = []

        image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")

        # Run inference args: https://docs.ultralytics.com/modes/predict/#inference-arguments
        #results = self.model(image_cv, verbose=False, conf=CONF_THRESH, imgsz=(736, 1280))[0] # do this for realsense (img dim not a multiple of max stride length 32)
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]

        # Extract segmentation masks
        if (results.boxes is None):
            return

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = PILImage.fromarray(im_array[..., ::-1])  # RGB PIL image
            
            msg_yolo_detections = RosImage()
            msg_yolo_detections.header.stamp = rgb.header.stamp
            msg_yolo_detections.height = im.height
            msg_yolo_detections.width = im.width
            msg_yolo_detections.encoding = "rgb8"
            msg_yolo_detections.is_bigendian = False
            msg_yolo_detections.step = 3 * im.width
            msg_yolo_detections.data = np.array(im).tobytes()
            self.img_pub.publish(msg_yolo_detections)
            # im.show()  # show image
            # im.save('results.jpg')  # save image

        class_ids = results.boxes.cls.data.cpu().numpy()
        bboxes = results.boxes.xywh.data.cpu().numpy()
        confs = results.boxes.conf.data.cpu().numpy()

        for class_id, bbox, conf in zip(class_ids, bboxes, confs):
            # ---- Object Vector ----
            object = ObjectVector()
            class_id = int(class_id)
            print(conf)
            print(class_id)
            obj_centroid = (bbox[0], bbox[1])  # x, y
            bearing = obj_centroid[0]/rgb.width * CAM_FOV - CAM_FOV/2

            print(obj_centroid)
            print(bearing)
            range = ping_to_range(depth, bearing)            


            # Unproject centroid to 3D
            if range: 
                x, y, z = unproject(obj_centroid[0], obj_centroid[1], range, K)
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
    bridge = CvBridge()
    CAM_FOV = 80 # degrees
    calib_file_loc = '/home/singhk/data/building_1_pool/bluerov_1080_cal.yaml'
    with open(calib_file_loc) as stream:
        cam_info = yaml.safe_load(stream)
    
    K = np.array(cam_info['camera_matrix']['data']).reshape(3,3)
    
    
    rospy.spin()