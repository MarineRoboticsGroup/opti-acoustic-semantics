
from io import BytesIO
import rospy
import message_filters
from semanticslam_ros.msg import ObjectsVector, ObjectVector, ObjectsVectorUncertainty, ObjectVectorUncertainty
from visualization_msgs.msg import Marker

class GraphSaver:
    """
    Saves the fullgraph to a file
    """

    def __init__(self) -> None:        
        fullgraph_topic = rospy.get_param("~fullgraph_topic", "/uncertainty_landmark_features")
        lm_colors_topic = rospy.get_param("~lm_colors_topic", "/landmarks")
        
        self.fullgraph_sub = message_filters.Subscriber(fullgraph_topic, ObjectsVectorUncertainty, queue_size=1)
        self.lm_colors_sub = message_filters.Subscriber(lm_colors_topic, Marker, queue_size=1)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.fullgraph_sub, self.lm_colors_sub), 1, 0.025
        )

        self.sync.registerCallback(self.write_color_map_to_file)
        # self.fullgraph_sub = rospy.Subscriber(fullgraph_topic, ObjectsVector, self.write_map_to_file)
    
    def write_map_to_file(self, fullgraph: ObjectsVector) -> None:
        buff = BytesIO()
        fullgraph.serialize(buff)
        serialized_bytes = buff.getvalue()
        with open("zpool_2obj2loop_part1.txt", "wb") as binary_file:
            # Write bytes to file
            binary_file.write(serialized_bytes)

    
    def write_color_map_to_file(self, fullgraph: ObjectsVectorUncertainty, lm_colors: Marker) -> None:
        buff = BytesIO()
        fullgraph.serialize(buff)
        serialized_bytes = buff.getvalue()
        with open("zpool_2obj2loop_part1_uncertainties.txt", "wb") as binary_file:
            # Write bytes to file
            binary_file.write(serialized_bytes)
            
        buff2 = BytesIO()
        lm_colors.serialize(buff2)
        serialized_bytes2 = buff2.getvalue()
        with open("zpool_2obj2loop_part1_colors_uncertainties.txt", "wb") as binary_file2:
            # Write bytes to file
            binary_file2.write(serialized_bytes2)
        
if __name__ == "__main__":
    rospy.init_node("graph_saver")
    graph_saver = GraphSaver()
    rospy.spin()