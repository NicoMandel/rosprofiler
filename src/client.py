#!/usr/bin/env python3


import rospy
from ros_statistics_msgs.msg import HostStatistics, NodeStatistics



class ProfileClient:

    def __init__(self):
        """ A client class on the rosprofiler topic msgs"""
        # Waiting for the topic to get going
        rospy.wait_for_message("/host_statistics", HostStatistics)

    
    def host_callback(self, msg):
        """ Callback on the host_statistics topic
         """
        pass

    
    def node_callback(self, msg):
        """ Callback on the node_statistics topic
        """

        pass










if __name__=="__main__":
    try:
        rospy.init_node("rosprofiler_client")
        cl = ProfileClient()
    except Exception as e:
        rospy.logerr(e)
    
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)

    