#!/usr/bin/env python3

import rospy
from rostopic import ROSTopicHz, ROSTopicBandwidth as ROSTopicBw
import rostopic
import os.path
import pandas as pd
from copy import deepcopy


class Timinglogger:

    def __init__(self, topic, metrics, window_size=100, filter_expr=None, timer_freq=0.5, filename="freqfile"):
        """ A topic logger class, which will initiate ROS BW and HZ loggers for the specified topic
        """
        # self.topic = topic.replace('/','')
        # self.topic = topic
        self.Metrics = metrics 

        self.rthz = ROSTopicHz(window_size, filter_expr=filter_expr)
        rospy.Subscriber(topic, rospy.AnyMsg, self.rthz.callback_hz)

        # Access the variables through "rt.times", see https://github.com/strawlab/ros_comm/blob/6f7ea2feeb3c890699518cb6eb3d33faa15c5306/tools/rostopic/src/rostopic.py#L136
        self.rtbw = ROSTopicBw() # Default window_size is 100
        rospy.Subscriber(topic, rospy.AnyMsg, self.rtbw.callback)

        # Timer to repeatedly log the values. Documentation here: http://docs.ros.org/melodic/api/rospy/html/rospy.timer.Timer-class.html
        self.timer = rospy.Timer(rospy.Duration(timer_freq), self.timer_callback)
        self.logging_df = pd.DataFrame(columns=self.Metrics).set_index("Time")

    def timer_callback(self, event): # has to take a rospy.TimerEvent, see documentation
        """
            Callback on the timing topic. This gets the data from BW and HZ objects
        """
        if len(self.rtbw.times or self.rthz.times) < 2:
            return
        else:
            temp_df = pd.DataFrame(columns=self.Metrics).set_index("Time")
            # temp_bw_df = pd.DataFrame(columns=self.BW_metrics).set_index("Time")

            # Section for the HZ stuff. Copy of the "print" section
            # TODO: do i need the with self.rthz.lock: in here
            hz_vals = pd.np.asarray(self.rthz.times)
            t = deepcopy(self.rthz.msg_tn)
            if t is None:
                t = rospy.get_rostime().to_sec()
            temp_df.at[t, "HZ_Samples"] = hz_vals.shape[0]
            temp_df.at[t, "HZ_Mean"] = 1. / pd.np.mean(hz_vals)
            temp_df.at[t, "HZ_Max_Delta"] = pd.np.max(hz_vals)
            temp_df.at[t, "HZ_Min_Delta"] = pd.np.min(hz_vals)
            temp_df.at[t, "HZ_Std Dev_Delta"] = pd.np.std(hz_vals)

            # section for the bw stuff. Copy of the "print" section
            bw_times = pd.np.asarray(self.rtbw.times)
            bw_sizes = pd.np.asarray(self.rtbw.sizes)
            total = pd.np.sum(bw_sizes)
            temp_df.at[t, "BW_Bytes / sec"] = total /(bw_times[-1] - bw_times[0])
            temp_df.at[t, "BW_Samples"] = bw_sizes.shape[0]
            temp_df.at[t, "BW_Mean"] = pd.np.mean(bw_sizes)
            temp_df.at[t, "BW_Max"] = pd.np.max(bw_sizes)
            temp_df.at[t, "BW_Min"] = pd.np.min(bw_sizes)

            self.logging_df = pd.concat([self.logging_df, temp_df], sort=False)
            # rospy.loginfo("Bottom end of the table for: {}".format(self.topic))
            # rospy.loginfo(self.logging_df.tail())

class LoggerList:

    def __init__(self, list_of_topics, update_rate, sample_rate):
        """ A Loggerlist object, which wraps many timing loggers. This also implements the writetoFile Funciton """
        self.filename = rospy.get_param("filename", default="default")
        self.metrics = ["Time", "HZ_Samples", "HZ_Mean", "HZ_Max_Delta", "HZ_Min_Delta", "HZ_Std Dev_Delta", "BW_Samples", "BW_Bytes / sec", "BW_Mean", "BW_Max", "BW_Min"]
        self.loglist = {}
        self.topic_list = list_of_topics
        all_topics = rospy.get_published_topics()          
        
        self.sample_rate =sample_rate
        all_ts = [name for name, _ in all_topics]
        # for topic, t_type in all_topics:
        #         all_ts.append(topic)
        for topic in self.topic_list:
            for name in all_ts:                   # Topic format see documentation in README
                if topic in name:
                    self.loglist[name]= Timinglogger(name, self.metrics, timer_freq=self.sample_rate)
                    rospy.loginfo("Logging enabled for topic: {}".format(name))

        # update timer - in case of delayed topics
        self.update_rate = rospy.Duration(update_rate)
        self._topic_timer = rospy.Timer(self.update_rate, self.update_topic_list)

        # Shutdown hook
        rospy.on_shutdown(self.writeToFile)

    def update_topic_list(self, event=None):
        """ 
        Function to update the topic list - in case of delayed topics
        """
        all_topics = rospy.get_published_topics()
        all_ts = [name for name, _ in all_topics]
        for topic in self.topic_list:
            for name in all_ts:
                if (topic in name) and (name not in self.loglist.keys()):
                    self.loglist[name] = Timinglogger(name, self.metrics, timer_freq=self.sample_rate)
                    rospy.loginfo("Logging enabled for topic: {}".format(name))
            

        
    def writeToFile(self):
        """
            Function to be called on shutdown. Stops timer and writes data to a file
        """
        parentDir = os.path.dirname(__file__)
        fname = os.path.abspath(os.path.join(parentDir, '..','results',self.filename+'_timing'+'.xlsx'))
        self._topic_timer.shutdown()
        with pd.ExcelWriter(fname) as writer:
            rospy.loginfo("Writing results to file: {}".format(fname))
            for key, Logger in self.loglist.items():
                # rospy.loginfo("logging results: {},: \n {}".format(key.replace('/',''), Logger.logging_df.tail()))
                if Logger.logging_df.shape[0] > 2:
                    Logger.logging_df.to_excel(writer, sheet_name=key.replace('/','_'))
                Logger.timer.shutdown()


if __name__=="__main__":
    try:
        rospy.init_node("Freq_logger")
        topics = rospy.get_param('topics', default=None)
        update_rate = rospy.get_param("updateRate", default=2)
        sample_rate = 0.5
        if topics is None:
            rospy.logwarn("No topics specified. Looking for All topics")
            ts = rospy.get_published_topics()   # this gives all published topics as a tuple with the types
            topics = []
            for topic, t_type in ts:
                topics.append(topic)
        # use the wrapper list holding object
        LL = LoggerList(topics, update_rate, sample_rate)
    except rospy.ROSInitException as e:
        rospy.logerr(e)

    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)