import rospy
import rosbag
from sensor_msgs.msg import Image as RosImage
import tf2_ros
#import tf2_geometry_msgs

import message_filters
from std_msgs.msg import Int32, Float32
from nav_msgs.msg import Odometry
from sonar_oculus.msg import OculusPing
from sensor_msgs.msg import CameraInfo

import yaml

count = 0

# Set the path for the ROS bag file
rosbag_filename = 'kelpie_filtered.bag'

# Initialize a ROS bag
out_bag = rosbag.Bag(rosbag_filename, 'w')


def sync_callback(image_msg, sonar_msg, odom_msg, gt_msg, sonar_img_msg):
    global count, out_bag
    print(count)
    print(image_msg.header.stamp)
    print(sonar_msg.header.stamp)
    print(odom_msg.header.stamp)
    print(gt_msg.header.stamp)
    print(sonar_img_msg.header.stamp)
    count += 1

    out_bag.write('/kelpie/usb_cam/image_raw_repub', image_msg, rospy.Time.now())
    out_bag.write('/sonar_oculus_node/M750d/ping', sonar_msg, rospy.Time.now())
    out_bag.write('/odometry/differential', odom_msg, rospy.Time.now())
    out_bag.write('/ground_truth', gt_msg, rospy.Time.now())
    out_bag.write('/sonar_oculus_node/M750d/image', sonar_img_msg, rospy.Time.now())
    


if __name__ == "__main__":
    rospy.init_node('kelpie_syncing_node')
    
    image_topic = "/kelpie/usb_cam/image_raw_repub"
    odom_topic = "/odometry/differential"
    gt_topic = "/ground_truth"
    sonar_topic = "/sonar_oculus_node/M750d/ping"
    sonar_image_topic = "/sonar_oculus_node/M750d/image"

    image_sub = message_filters.Subscriber(image_topic, RosImage)
    sonar_sub = message_filters.Subscriber(sonar_topic, OculusPing)
    odom_sub = message_filters.Subscriber(odom_topic, Odometry)
    gt_sub = message_filters.Subscriber(gt_topic, Odometry)
    sonar_img_sub = message_filters.Subscriber(sonar_image_topic, RosImage)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, sonar_sub, odom_sub, gt_sub, sonar_img_sub], 100, 100, allow_headerless=False)
    ts.registerCallback(sync_callback)

    print("STARTED SUBSCRIBER!")


    rospy.spin()
    out_bag.close()