#!/usr/bin/env python3

""" 
TODO:
    1. Check what the bloody rosprofiler actually records - averages, times, whatever
        1. Check the units
        2. Check at which intervals / timeframes it samples and deletes / resets
        3. 
    2. Create a toy example with the listener / talker that we wrote
    3. record a toy example with time information
    4. extend the class with:
        1. bw and hz from the documentation I found online. List which Topics this should be done for
        2. for the Jetson Nano: get the values for the Power consumption
    5. Put the values into relation with the ISO standard and what they need
"""


import rospy
from ros_statistics_msgs.msg import HostStatistics, NodeStatistics



class ProfileClient:

    def __init__(self):
        """ A client class on the rosprofiler topic msgs"""
        # Waiting for the topic to get going
        rospy.wait_for_message("/host_statistics", HostStatistics)

        # Subscribers
        rospy.Subscriber("/host_statistics", HostStatistics, self.host_callback, queue_size=10)
        rospy.Subscriber("/node_statistics", NodeStatistics, self.node_callback, queue_size=10)

        # TODO 3: turn the nodes and hosts which are supposed to be listened to into ROS parameters
        # https://www.geeksforgeeks.org/display-hostname-ip-address-python/
        # for now: hardcoded lists
        ip_nano = "192.168.1.120"
        self.ips = [ip_nano]

        node="wp_node"
        self.nodes[node]

    
    def host_callback(self, msg):
        """ Callback on the host_statistics topic:
            hostname, ipaddress, window start and stop times,
            sample number
            cpu load: mean, std dev, max
            phymem_used: mean, std, max
            phymem_avail: mean, std, max
            TODO: Check in the psutil documentation what these values actually mean
            TODO: Check in the rosprofiler nodes how and what this records. does this clear before resampling or take historical average?
        """

        for ip in self.ips:
            if msg.ipadress == ip:
                # TODO 1: get the values out here
                pass
        

    
    def node_callback(self, msg):
        """ Callback on the node_statistics topic. Given Values are:
        Threads, Cpu load, virtual and real memory
        """

        for node in self.nodes:
            if msg.node == node:
                # TODO 2: Node matched. Do stuff here
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

    