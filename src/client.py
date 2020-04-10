#!/usr/bin/env python3

import rospy
from ros_statistics_msgs.msg import NanoStatistics, NodeStatisticsNano
import socket
import pandas as pd # to gather the data collected
import os.path
import rosnode


class ProfileClient:

    def __init__(self):
        """ A client class on the rosprofiler topic msgs"""

        self.filename = rospy.get_param('filename', default="default")
        # Get parameters from server to find out what to track and where to write it
        hosts = rospy.get_param('hosts', default=None)
        self.nodes = rospy.get_param('nodes', default=None)
        if hosts is None:
            rospy.logwarn("No Machines specified. Logging All")
            hosts = rosnode.get_machines_by_nodes()

        if self.nodes is None:
            rospy.logwarn("No Nodes specified. Logging All")
            self.nodes = rosnode.get_node_names()

        # Setup work for the hosts
        self.extracted_statistics_host = ["Time", "Duration", "Samples", "CPU Count", "Power"
        "CPU Load mean", "CPU Load max", "CPU Load std"
        "Used Memory mean", "Used Memory max", "Used Memory std"
        "Available Memory mean", "Available Memory min", "Available Memory std", 
        "Shared Memory mean", "Shared Memory std", "Shared Memory max",
        "Swap Available mean", "Swap Available std", "Swap Available min",
        "Swap Used mean", "Swap Used std", "Swap Used max"        
        ]
        host_df = pd.DataFrame(columns=self.extracted_statistics_host).set_index("Time")

        # Assign a Dataframe to each host 
        self.ips = []
        self.host_df_dict = {}
        for host in hosts:
            ip = socket.gethostbyname(host)
            self.ips.append(ip)
            self.host_df_dict[ip] = host_df.copy(deep=True)
        
        # Setup work for the Nodes
        self.extracted_statistics_node = ["Time", "Duration", "Samples", "CPU Count"
        "Threads", "CPU Load mean", "CPU Load max", "CPU Load std"
        "PSS mean", "PSS std", "PSS max"
        "Swap Used mean", "Swap Used std", "Swap Used max",
        "Virtual Memory mean", "Virtual Memory std", "Virtual Memory max"
        ]

        node_df = pd.DataFrame(columns=self.extracted_statistics_node).set_index("Time")
        self.node_df_dict= {}
        for node in self.nodes:
            self.node_df_dict[node] = node_df.copy(deep=True)
        
        
        # Waiting for the topic to get going and setting up shutdown function
        rospy.wait_for_message("host_statistics", NanoStatistics)
        rospy.on_shutdown(self.writeToFile)

        # Subscribers last - to not mess up when messages come in before everything is set up
        rospy.Subscriber("host_statistics", NanoStatistics, self.host_callback, queue_size=10)
        rospy.Subscriber("node_statistics", NodeStatisticsNano, self.node_callback, queue_size=10)

    
    def host_callback(self, msg):
        """ Callback on the host_statistics topic:
            hostname, ipaddress, window start and stop times,
            cpu_count 
            sample number
            power consumption

            cpu load: mean, std dev, max
            
            phymem_used: mean, std, max
            phymem_avail: mean, std, min
            phymem_shared: mean, std, max

            swap_used: mean, std, max
            swap_available: mean, std, min
        """

        if msg.ipaddress in self.ips:
            temp_df = pd.DataFrame(columns=self.extracted_statistics_host).set_index("Time")

            # Times converted to milisecond durations
            t = msg.window_stop.to_sec()
            duration = (msg.window_stop - msg.window_start).to_nsec() / 1000000
            temp_df.at[t, "Duration"] = duration
            temp_df.at[t, "Samples"] = float(msg.samples)
            temp_df.at[t, "CPU Count"] = msg.cpu_count

            # Power is smaller than Zero if its not a Nano
            temp_df.at[t, "Power"] = msg.power

            # CPU statistics 
            temp_df.at[t, "CPU Load mean"] = msg.cpu_load_mean
            temp_df.at[t, "CPU Load max"] = msg.cpu_load_max
            temp_df.at[t, "CPU Load std"] = msg.cpu_load_std

            # Memory statistics - Used, Available and shared 
            temp_df.at[t, "Used Memory mean"] = (int(pd.np.floor(msg.phymem_used_mean)) >> 20)
            temp_df.at[t, "Used Memory max"] = (int(pd.np.floor(msg.phymem_used_max)) >> 20)
            temp_df.at[t, "Used Memory std"] = (int(pd.np.floor(msg.phymem_used_std)) >> 20)
            temp_df.at[t, "Available Memory mean"] = (int(pd.np.floor(msg.phymem_avail_mean)) >> 20)
            temp_df.at[t, "Available Memory min"] = (int(pd.np.floor(msg.phymem_avail_min)) >> 20)
            temp_df.at[t, "Available Memory std"] = (int(pd.np.floor(msg.phymem_avail_std)) >> 20) # TODO: KBs?
            temp_df.at[t, "Shared Memory mean"] = (int(pd.np.floor(msg.phymem_shared_mean)) >> 20)
            temp_df.at[t, "Shared Memory mean"] = (int(pd.np.floor(msg.phymem_shared_std)) >> 20)
            temp_df.at[t, "Shared Memory mean"] = (int(pd.np.floor(msg.phymem_shared_max)) >> 20)

            # Memory statistics - Swap - used and available
            temp_df.at[t, "Swap Used mean"] = (int(pd.np.floor(msg.swap_used_mean)) >> 20)
            temp_df.at[t, "Swap Used std"] = (int(pd.np.floor(msg.swap_used_std)) >> 20)
            temp_df.at[t, "Swap Used max"] = (int(pd.np.floor(msg.swap_used_max)) >> 20)
            temp_df.at[t, "Swap Available mean"] = (int(pd.np.floor(msg.swap_avail_mean)) >> 20)
            temp_df.at[t, "Swap Available std"] = (int(pd.np.floor(msg.swap_avail_std)) >> 20)
            temp_df.at[t, "Swap Available min"] = (int(pd.np.floor(msg.swap_avail_min)) >> 20)

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
            temp_df.at[t, "Samples"] = msg.samples

            # CPU Stuff
            temp_df.at[t, "Threads"] = msg.threads
            temp_df.at[t, "CPU Count"] = msg.cpu_count
            temp_df.at[t, "CPU Load mean"] = msg.cpu_load_mean 
            temp_df.at[t, "CPU Load max"] = msg.cpu_load_max
            temp_df.at[t, "CPU Load std"] = msg.cpu_load_std
            
            # Memory - Virtual, PSS and Swap
            temp_df.at[t, "Virtual Memory mean"] = (int(pd.np.floor(msg.virt_mem_mean)) >> 20)
            temp_df.at[t, "Virtual Memory std"] = (int(pd.np.floor(msg.virt_mem_std ))>> 20)
            temp_df.at[t, "Virtual Memory max"] = (int(pd.np.floor(msg.virt_mem_max ))>> 20)
            temp_df.at[t, "PSS mean"] = (int(pd.np.floor(msg.pss_mean)) >> 20)
            temp_df.at[t, "PSS std"] = (int(pd.np.floor(msg.pss_std))>> 20)
            temp_df.at[t, "PSS max"] = (int(pd.np.floor(msg.pss_max))>> 20)

            temp_df.at[t, "Swap Used mean"] = (int(pd.np.floor(msg.swap_mean))>> 20)
            temp_df.at[t, "Swap Used std"] = (int(pd.np.floor(msg.swap_std))>> 20)
            temp_df.at[t, "Swap Used max"] = (int(pd.np.floor(msg.swap_max))>> 20)
            
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
    except rospy.ROSInitException as e:
        rospy.logerr(e)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)

    