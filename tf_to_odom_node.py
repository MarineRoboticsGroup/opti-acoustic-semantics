#!/home/singhk/miniconda3/envs/dino-vit-feats-env/bin python

import rospy 
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from tf.transformations import *
import numpy as np

# per frame at ~238 hz (mocap rate)
# with multiplier == 1, this is 0.00238 m/s,  2cm drift in each direction every 10 seconds 
# added_noise_multiplier = 5
# added_noise_sigmas = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001] # x, y, z, roll, pitch, yaw
# odom_noise = added_noise_multiplier * added_noise_sigmas

# added_noise_gaussian = [np.random.normal(0, odom_noise[i], 1) for i in range(6)]

tf_pub = rospy.Publisher("/pose", Odometry, queue_size=10)

def tf_callback(tf_msg):
    odom_msg = Odometry()
    odom_msg.header = tf_msg.transforms[0].header # assuming only transform is mocap to camera
    odom_msg.child_frame_id = tf_msg.transforms[0].child_frame_id # not actually used by maxmixtures
    
    # euler = euler_from_quaternion([tf_msg.transforms[0].transform.rotation.x, tf_msg.transforms[0].transform.rotation.y, tf_msg.transforms[0].transform.rotation.z, tf_msg.transforms[0].transform.rotation.w])
    # transform = [tf_msg.transforms[0].transform.translation.x, tf_msg.transforms[0].transform.translation.y, tf_msg.transforms[0].transform.translation.z, euler[0], euler[1], euler[2]]
    # noisy_tf = [transform[i] + added_noise_gaussian[i] for i in range(len(transform))]
    
    
    # noisy_quat = quaternion_from_euler(noisy_tf[3], noisy_tf[4], noisy_tf[5])
    
    odom_msg.pose.pose.position.x = tf_msg.transforms[0].transform.translation.x
    odom_msg.pose.pose.position.y = tf_msg.transforms[0].transform.translation.y
    odom_msg.pose.pose.position.z = tf_msg.transforms[0].transform.translation.z
    odom_msg.pose.pose.orientation.x = tf_msg.transforms[0].transform.rotation.x
    odom_msg.pose.pose.orientation.y = tf_msg.transforms[0].transform.rotation.y
    odom_msg.pose.pose.orientation.z = tf_msg.transforms[0].transform.rotation.z
    odom_msg.pose.pose.orientation.w = tf_msg.transforms[0].transform.rotation.w
    
    tf_pub.publish(odom_msg)


if __name__ == "__main__":
    rospy.init_node('tf_to_odom_node')
    rospy.Subscriber("/tf", TFMessage, tf_callback)
    rospy.spin()