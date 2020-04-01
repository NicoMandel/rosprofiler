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

    TODO: Convert the cpu_load msgs, they are arrays


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
        # for now: hardcoded lists
        ip_nano = socket.gethostbyname("nano-wired")
        self.ips = [ip_nano]

        node="wp_node"
        self.nodes = [node]
        self.extracted_statistics_node = ["Duration", "Samples", "Threads", "CPU load mean", "CPU load max", "Virtual Memory mean", "Virtual memory Max", "Real Memory Mean", "Real Memory Max"]
        self.node_df = pd.DataFrame(index=extracted_statistics_node, columns=self.nodes)

        # Create a temporary dataframe, for once-off use
        self._node_reset_array = pd.np.zeros((1, len(self.extracted_statistics_node)),dtype=pd.np.float64)
        self._node_temp_df = pd.DataFrame(self._node_reset_array,columns=self.extracted_statistics_node)
        
        # Do the same for the hosts
        self.extracted_statistics_host = ["Duration" "Samples", "CPU load mean", "CPU load max", "Phymem used mean", "Phymem used max", "phymem avail mean", "Phymem avail max"]
        self.host_df = pd.DataFrame(index=extracted_statistics_host, columns=self.ips)
        self._host_reset_arr = pd.np.zeros((1, len(self.extracted_statistics_host)), dtype=pd.np.float64)
        self._host_temp_df = pd.DataFrame(self._host_reset_arr,columns=self.extracted_statistics_host)

    
    def host_callback(self, msg):
        """ Callback on the host_statistics topic:
            hostname, ipaddress, window start and stop times,
            sample number
            cpu load: mean, std dev, max
            phymem_used: mean, std, max
            phymem_avail: mean, std, max
        """

        temp_df = self._host_temp_df.copy(deep=True)
        # Times converted to milisecond durations
        duration = (rospy.Duration(msg.window_stop - msg.window_start).to_nsec()) / 1000
        temp_df.at[0, "Duration"] = duration
        temp_df.at[0, "Samples"] = float(msg.samples)
        # TODO: Convert these, they are arrays
        temp_df.at[0, "CPU load mean"] = msg.cpu_load_mean
        temp_df.at[0, "CPU load max"] = msg.cpu_load_max
        # TODO: bit shift these values
        temp_df.at[0, "Phymem used mean"] = msg.phymem_used_mean
        temp_df.at[0, "Phymem used max"] = msg.phymem_used_max
        temp_df.at[0, "phymem avail mean"] = msg.phymem_avail_mean
        temp_df.at[0, "Phymem avail max"] = msg.phymem_avail_max

        target_df = self.host_df_dict[msg.ipaddress]
        
        self.host_df_dict[msg.ipaddress] = self.concat_df(target_df, temp_df)


        
    def node_callback(self, msg):
        """ Callback on the node_statistics topic. Given Values are:
        Threads, Cpu load, virtual and real memory
        """
        # 0. initialise an empty dataframe
        temp_df = self._node_temp_df.copy(deep=True)
        
        pd.DataFrame(self._node_reset_array,columns=self.extracted_statistics_node)
        # 1. Get all the values of interest out
        # Times converted to milisecond durations
        duration = (rospy.Duration(msg.window_stop - msg.window_start).to_nsec()) / 1000 
        temp_df.at[0, "Duration"] = duration
        temp_df.at[0, "Samples"] = float(msg.samples)
        temp_df.at[0, "Threads"] = float(msg.threads)
        # TODO: percentage of total total local use - change this to be more meaningful from psutil
        temp_df.at[0, "CPU load mean"] = msg.cpu_load_mean 
        temp_df.at[0, "CPU load max"] = msg.cpu_load_max
        temp_df.at[0, "Virtual Memory mean"] = msg.virt_mem_mean
        temp_df.at[0, "Virtual memory Max"] = msg.virt_mem_max
        temp_df.at[0, "Real Memory Mean"] = msg.real_mem_mean
        temp_df.at[0, "Real Memory Max"] = msg.real_mem_max
        
        # 2. Get the target dataframe which the values should be appended to out
        target_df = self.node_df_dict[msg.node]

        # 3. Concatenate the dfs
        self.node_df_dict[msg.node] = self.concat_df(target_df, temp_df)


    # Helper functions        
    concat_df = staticmethod(lambda full_df, temp_df: pd.concat([full_df, temp_df]))

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

    