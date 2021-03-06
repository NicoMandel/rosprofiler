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
import rosnode


class ProfileClient:

    def __init__(self, fname="default"):
        """ A client class on the rosprofiler topic msgs"""

        # Get parameters from server to find out which files to track
        hosts = rospy.get_param('/hosts', default=None)
        self.nodes = rospy.get_param('/nodes', default=None)
        if hosts is None:
            rospy.logwarn("No Machines specified. Logging All")
            hosts = rosnode.get_machines_by_nodes()

        if self.nodes is None:
            rospy.logwarn("No Nodes specified. Logging All")
            self.nodes = rosnode.get_node_names()

        # Setup work for the hosts
        self.extracted_statistics_host = ["Time","Duration", "Samples", "CPU load mean of max", "CPU load max of mean", "Phymem used mean", "Phymem used max", "phymem avail mean", "Phymem avail max"]
        host_df = pd.DataFrame(columns=self.extracted_statistics_host).set_index("Time")

        # Assign a Dataframe to each host 
        self.ips = []
        self.host_df_dict = {}
        for host in hosts:
            ip = socket.gethostbyname(host)
            self.ips.append(ip)
            self.host_df_dict[ip] = host_df.copy(deep=True)
        
        # Setup work for the Nodes
        self.extracted_statistics_node = ["Time", "Duration", "Samples", "Threads", "CPU load mean", "CPU load max", "Virtual Memory mean", "Virtual memory Max", "Real Memory Mean", "Real Memory Max"]
        node_df = pd.DataFrame(columns=self.extracted_statistics_node).set_index("Time")
        self.node_df_dict= {}
        for node in self.nodes:
            self.node_df_dict[node] = node_df.copy(deep=True)
        
        
        # Waiting for the topic to get going and setting up shutdown function
        rospy.wait_for_message("/host_statistics", HostStatistics)
        self.filename = fname
        rospy.on_shutdown(self.writeToFile)

        # Subscribers last - to not mess up when messages come in before everything is set up
        rospy.Subscriber("/host_statistics", HostStatistics, self.host_callback, queue_size=10)
        rospy.Subscriber("/node_statistics", NodeStatistics, self.node_callback, queue_size=10)

    
    def host_callback(self, msg):
        """ Callback on the host_statistics topic:
            hostname, ipaddress, window start and stop times,
            sample number
            cpu load: mean, std dev, max
            phymem_used: mean, std, max
            phymem_avail: mean, std, max
        """
        if msg.ipaddress in self.ips:
            temp_df = pd.DataFrame(columns=self.extracted_statistics_host).set_index("Time")

            # Times converted to milisecond durations
            t = msg.window_stop.to_sec()
            duration = (msg.window_stop - msg.window_start).to_nsec() / 1000000
            temp_df.at[t, "Duration"] = duration
            temp_df.at[t, "Samples"] = float(msg.samples)
            # Converted host statistics - mean of max and max of mean
            temp_df.at[t, "CPU load max of mean"] = pd.np.mean(msg.cpu_load_mean)
            temp_df.at[t, "CPU load mean of max"] = pd.np.max(msg.cpu_load_max)
            temp_df.at[t, "Phymem used mean"] = (int(pd.np.floor(msg.phymem_used_mean)) >> 20)
            temp_df.at[t, "Phymem used max"] = (int(pd.np.floor(msg.phymem_used_max)) >> 20)
            temp_df.at[t, "phymem avail mean"] = (int(pd.np.floor(msg.phymem_avail_mean)) >> 20)
            temp_df.at[t, "Phymem avail max"] = (int(pd.np.floor(msg.phymem_avail_max)) >> 20)
            target_df = self.host_df_dict[msg.ipaddress]
            self.host_df_dict[msg.ipaddress] = self.concat_df(target_df, temp_df)

       
    def node_callback(self, msg):
        """ Callback on the node_statistics topic. Given Values are:
            Threads, Cpu load, virtual and real memory
        """

        if msg.node in self.nodes:
            # 0. initialise an empty dataframe
            temp_df = pd.DataFrame(columns=self.extracted_statistics_node).set_index("Time")
            
            # 1. Get all the values of interest out
            # Times converted to second durations
            t = msg.window_stop.to_sec()
            duration = (msg.window_stop - msg.window_start).to_nsec() / 1000000 
            temp_df.at[t, "Duration"] = duration
            temp_df.at[t, "Samples"] = float(msg.samples)
            temp_df.at[t, "Threads"] = float(msg.threads)
            # TODO: percentage of total total local use - change this to be more meaningful from psutil
            temp_df.at[t, "CPU load mean"] = msg.cpu_load_mean 
            temp_df.at[t, "CPU load max"] = msg.cpu_load_max
            temp_df.at[t, "Virtual Memory mean"] = (int(pd.np.floor(msg.virt_mem_mean)) >> 20)
            temp_df.at[t, "Virtual memory Max"] = (int(pd.np.floor(msg.virt_mem_max ))>> 20)
            temp_df.at[t, "Real Memory Mean"] = (int(pd.np.floor(msg.real_mem_mean)) >> 20)
            temp_df.at[t, "Real Memory Max"] = (int(pd.np.floor(msg.real_mem_max ))>> 20)
            
            # 2. Get the target dataframe which the values should be appended to out
            target_df = self.node_df_dict[msg.node]

            # 3. Concatenate the dfs
            self.node_df_dict[msg.node] = self.concat_df(target_df, temp_df)


    # Helper functions        
    concat_df = staticmethod(lambda full_df, temp_df: pd.concat([full_df, temp_df]))

    def writeToFile(self):
        """
        Function to record all the collected data into an Excel file - use pd.excelwriter
        """

        parentDir = os.path.dirname(__file__)
        fname = os.path.abspath(os.path.join(parentDir, '..','results',self.filename+'_nodes'+'.xlsx'))
        with pd.ExcelWriter(fname) as writer:
            print("Writing to file: {}".format(fname))
            for node_name, df in self.node_df_dict.items():
                if df.shape[0] > 2:
                    df.to_excel(writer, sheet_name=node_name.replace('/',''))

        fname =  os.path.abspath(os.path.join(parentDir, '..','results',self.filename+'_hosts'+'.xlsx'))
        with pd.ExcelWriter(fname) as writer:
            print("Writing to file: {}".format(fname))
            for host_name, df in self.host_df_dict.items():
                if df.shape[0] > 2:
                    df.to_excel(writer, sheet_name=host_name)
            


if __name__=="__main__":
    try:
        rospy.init_node("rosprofiler_client")
        cl = ProfileClient()
    except rospy.ROSException as e:
        rospy.logerr(e)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)

    