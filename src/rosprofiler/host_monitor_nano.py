# Copyright 2014 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing  # for multiprocessing.cpu_count()
import numpy as np
import psutil

import rosgraph
import rospy
import socket

from ros_statistics_msgs.msg import NanoStatistics


class HostMonitor(object):
    """ Tracks cpu and memory information of the host. """
    def __init__(self):
        self._hostname = rosgraph.network.get_host_name()
        self._ipaddress = rosgraph.network.get_local_address()
        self._cpus_available = multiprocessing.cpu_count()

        self.cpu_load_log = list()
        self.phymem_used_log = list()
        self.phymem_avail_log = list()
        self.swap_used_log = list()
        self.swap_avail_log = list()
        self.start_time = rospy.get_rostime()

        # Section on defining the Nano filename
        sys_host = socket.gethostname()
        if "nan" in sys_host.lower():
            self.power_file = "/sys/bus/i2c/devices/6-0040/iio_device/in_power0_input"
        else:
            self.power_file = ""

        # cpu count
        self._cpu_count = psutil.cpu_count()

    def update(self):
        """ Record information about the cpu and memory usage for this host into a buffer """
        self.cpu_load_log.append(psutil.cpu_percent(interval=None, percpu=False))
        self.phymem_used_log.append(psutil.virtual_memory().used)
        self.phymem_avail_log.append(psutil.virtual_memory().available)
        self.swap_avail_log.append(psutil.swap_memory().free)
        self.swap_used_log.append(psutil.swap_memory().used)

    def get_statistics(self):
        """ Returns NanoStatistics() using buffered information.
        :returns: statistics information collected about the host
        :rtype: NanoStatistics
        """

        statistics = NanoStatistics()
        statistics.hostname = self._hostname
        statistics.ipaddress = self._ipaddress
        statistics.window_start = self.start_time
        statistics.window_stop = rospy.get_rostime()
        statistics.samples = len(self.cpu_load_log)

        # Section on getting the Nano power statistics
        try:
            with open(self.power_file, 'r') as pow_file:
                val = int(pow_file.read())
        except IOError:
            val = -1
        statistics.power = val
        statistics.cpu_count = self._cpu_count

        if len(self.cpu_load_log) > 0:
            cpu_load_log = np.array(self.cpu_load_log)
            statistics.cpu_load_mean = np.mean(cpu_load_log)
            statistics.cpu_load_std = np.std(cpu_load_log)
            statistics.cpu_load_max = np.max(cpu_load_log)
        if len(self.phymem_used_log) > 0:
            phymem_used_log = np.array(self.phymem_used_log)
            statistics.phymem_used_mean = np.mean(phymem_used_log)
            statistics.phymem_used_std = np.std(phymem_used_log)
            statistics.phymem_used_max = np.max(phymem_used_log)
        if len(self.phymem_avail_log) > 0:
            phymem_avail_log = np.array(self.phymem_avail_log)
            statistics.phymem_avail_mean = np.mean(phymem_avail_log)
            statistics.phymem_avail_std = np.std(phymem_avail_log)
            statistics.phymem_avail_max = np.max(phymem_avail_log)
        if len(self.swap_avail_log) > 0:
            swap_avail_log = np.array(self.swap_avail_log)
            statistics.swap_avail_mean = np.mean(swap_avail_log)
            statistics.swap_avail_std = np.std(swap_avail_log)
            statistics.swap_avail_min = np.min(swap_avail_log)
        if len(self.swap_used_log) > 0: 
            swap_used_log = np.array(self.swap_used_log)
            statistics.swap_used_mean = np.mean(swap_used_log)
            statistics.swap_used_std = np.std(swap_used_log)
            statistics.swap_used_max = np.max(swap_used_log)
        return statistics

    def reset(self):
        """ Clears the information buffer. """
        self.cpu_load_log = list()
        self.phymem_used_log = list()
        self.phymem_avail_log = list()
        self.swap_used_log = list()
        self.swap_avail_log = list()
        self.start_time = rospy.get_rostime()
