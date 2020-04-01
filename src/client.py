#!/usr/bin/env python3

""" 
TODO:
    1. Check what the bloody rosprofiler actually records - averages, times, whatever
        1. Check the units. Here: https://stackoverflow.com/questions/21792655/psutil-virtual-memory-units-of-measurement
        2. Check at which intervals / timeframes it samples and deletes / resets
        3. Check what the Memory units are - documented here: https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info
    2. Create a toy example with the listener / talker that we wrote
    3. record a toy example with time information
    4. extend the class with:
        1. bw and hz from the documentation I found online. List which Topics this should be done for
        2. for the Jetson Nano: get the values for the Power consumption
    5. Put the values into relation with the ISO standard and what they need


HINTS:  ROSNODE API to find lists and machines and nodes
http://docs.ros.org/hydro/api/rosnode/html/
"""


import rospy
from ros_statistics_msgs.msg import HostStatistics, NodeStatistics
import socket
import pandas as pd # to gather the data collected
import os.path


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
        ip_nano = socket.gethostbyname("nano-wired")
        self.ips = [ip_nano]

        node="wp_node"
        self.nodes = [node]
        self.extracted_statistics_node = ["Window Start","Window Stop", "Samples", "Threads", "CPU load mean", "CPU load max", "Virtual Memory mean", "Virtual memory Max", "Real Memory Mean", "Real Memory Max"]
        self.node_df = pd.DataFrame(index=extracted_statistics_node, columns=self.nodes)

        # Create a temporary dataframe, for once-off use
        self._temp_node_df = pd.DataFrame(0,columns=extracted_statistics_node)
        
        self.extracted_statistics_host = ["Window Start","Window Stop", "Samples", "CPU load mean", "CPU load max", "Phymem used mean", "Phymem used max", "phymem avail mean", "Phymem avail max"]
        self.host_df = pd.DataFrame(index=extracted_statistics_host, columns=self.ips)

    
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
                start_t = msg.window_start
                stop_t = msg.window_stop
                samples = msg.samples
                
                
        
    def node_callback(self, msg):
        """ Callback on the node_statistics topic. Given Values are:
        Threads, Cpu load, virtual and real memory
        """

        for node in self.nodes:
            if msg.node == node:
                # TODO 2: Node matched. Do stuff here
                pass
        
    
    def writeToFile(self, filename="Default"):
        """
        Function to record all the collected data into an Excel file - use pd.excelwriter
        TODO: Write this function
        """

        dirname = os.path.abspath(__file__)
        fname = os.path.join(dirname,fname+".xlsx")
        # with pd.ExcelWriter(fname_excel) as writer:
        #     df.to_excel(writer, sheet_name=target_f_name)





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
            # TODO: Write a function in here that will write everything from the class to file
            cl.writeToFile()
            rospy.logerr_once(e)

    