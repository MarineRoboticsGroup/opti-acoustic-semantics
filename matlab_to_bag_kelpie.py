import rosbag
import rospy
from nav_msgs.msg import Odometry
import tf


import numpy as np
import scipy.io as sio

# Set the path for the MATLAB data file
mat_filename = '/home/singhk/data/whoi_bluerov/processed_acoustics/IROS23.mat'
mat_data = sio.loadmat(mat_filename)


# Set the path for the ROS bag file
rosbag_filename = 'kelpie_acoustics.bag'

# Initialize a ROS bag
with rosbag.Bag(rosbag_filename, 'w') as bag:
    # Get timestamps array
    timestamps = mat_data['t'][0] # array dims are weird due to matlab 
    lbl_pos = mat_data['lbl_positions']
    lbl_orientations = mat_data['lbl_orientations']
    # lbl_pos_std_devs = mat_data['lbl_pos_std_devs']
    # lbl_theta_std_devs = mat_data['lbl_theta_std_devs']
    # lbl_covs = mat_data['lbl_covariances']
    gt_xyt = mat_data['XYT']

    # Iterate over the elements in the array corresponding to the current key
    for timestamp, pos, orientation, xyt in zip(timestamps, lbl_pos, lbl_orientations, gt_xyt):
        # Create a ROS message lbl + IMU odometry
        # x, y, yaw come from lbl. z, roll, pitch come from IMU
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.from_sec(float(timestamp))
        odom_msg.pose.pose.position.x = pos[0]
        odom_msg.pose.pose.position.y = pos[1]
        odom_msg.pose.pose.position.z = pos[2]
        
        odom_quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        odom_msg.pose.pose.orientation.x = odom_quaternion[0]
        odom_msg.pose.pose.orientation.y = odom_quaternion[1]
        odom_msg.pose.pose.orientation.z = odom_quaternion[2]
        odom_msg.pose.pose.orientation.w = odom_quaternion[3]
        
        # Append the message to the ROS bag with the corresponding timestamp
        bag.write('/odometry/differential', odom_msg, rospy.Time.from_sec(float(timestamp)))
        
        # create a ROS message for the ICP based positions 
        gt_msg = Odometry()
        gt_msg.header.stamp = rospy.Time.from_sec(float(timestamp))
        gt_msg.pose.pose.position.x = xyt[0]
        gt_msg.pose.pose.position.y = xyt[1]
        gt_msg.pose.pose.position.z = 0
        
        gt_quaternion = tf.transformations.quaternion_from_euler(0, 0, xyt[2])
        gt_msg.pose.pose.orientation.x = gt_quaternion[0]
        gt_msg.pose.pose.orientation.y = gt_quaternion[1]
        gt_msg.pose.pose.orientation.z = gt_quaternion[2]
        gt_msg.pose.pose.orientation.w = gt_quaternion[3]
        
        # Append the message to the ROS bag with the corresponding timestamp
        bag.write('/ground_truth', gt_msg, rospy.Time.from_sec(float(timestamp)))
        
        

print(f'ROS bag saved to {rosbag_filename}')