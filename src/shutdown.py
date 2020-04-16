#!/usr/bin/env python3

""" 
Program to run and wait for a signal. When the signal comes, it sends out a rospy.shutdown_signal

"""
import rospy
from std_msgs.msg import Bool

class signalHandler(object):

    def __init__(self, topic="/shutdown_signal"):
        super().__init__()
        rospy.init_node("Shutdown_node")
        rospy.loginfo("Listening for shutdown signal on topic: {}".format(topic))
        rospy.Subscriber(topic, Bool, callback=self.callback, queue_size=10)


    def callback(self, msg):
        """
        Callback on the shutdown msg 
        """
        rospy.loginfo("Initializing shutdown")
        rospy.signal_shutdown("Received shutdown Signal at {}".format(rospy.Time.now().to_sec()))


if __name__=="__main__":
    try:
        signalHandler()
    except rospy.ROSInitException as e:
        rospy.logerr(e)
    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)