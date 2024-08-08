
from io import BytesIO
import rospy
import message_filters
from semanticslam_ros.msg import ObjectsVector, ObjectVector

class GraphSaver:
    """
    Saves the fullgraph to a file
    """

    def __init__(self) -> None:        
        #subgraph_topic = rospy.get_param("~subgraph_topic", "/landmark_features")
        fullgraph_topic = rospy.get_param("~fullgraph_topic", "/landmark_features")
        
        # self.subgraph_sub = message_filters.Subscriber(subgraph_topic, ObjectsVector, queue_size=1)
        # self.fullgraph_sub = message_filters.Subscriber(fullgraph_topic, ObjectsVector, queue_size=1)
        # self.sync = message_filters.ApproximateTimeSynchronizer(
        #     (self.subgraph_sub, self.fullgraph_sub), 1, 0.025
        # )

        # self.sync.registerCallback(self.write_map_to_file)
        self.fullgraph_sub = rospy.Subscriber(fullgraph_topic, ObjectsVector, self.write_map_to_file)
    
    def write_map_to_file(self, fullgraph: ObjectsVector) -> None:
        buff = BytesIO()
        fullgraph.serialize(buff)
        serialized_bytes = buff.getvalue()
        with open("kitti_map.txt", "wb") as binary_file:
            # Write bytes to file
            binary_file.write(serialized_bytes)

        # p2.deserialize(serialized_bytes)
        
if __name__ == "__main__":
    rospy.init_node("graph_saver")
    graph_saver = GraphSaver()
    rospy.spin()