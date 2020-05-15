rosprofiler
===========

# Updates by Nico
* Changed [node_monitor.py](./src/rosprofiler/node_monitor.py) according to the psutil documentation
    * included correct named tuple referencing in the `update` function - mem_info

* Added a [client.py file](./src/client.py)
    * base structure is present
    * Depends on parameters to be set elsewhere which say what to log, otherwise will log everything by default, see [here](./src/client.py#L39)


## Nico TODOs:
[x] Fix the local shit - why are there no nodes being logged?
    * [x] is the [list comprehension](.src/profiler_Nano.py#L151) shitty? Tested [here](./src/teststuff.py#L23)
    * [x] is the [cross checking](./src/profiler_Nano.py#L100) just as terrible??
    * [x] **RUN** the Profiler file in debug mode

* [x]  Why the talker node does not show up in the log - assuming because of the ip and hostname confusion, as before, now appearing in [client.py](./src/client.py#L131) and [line ](./src/client.py#L54) - make consistent with the solution from [here](./src/rosprofiler/profiler_Nano.py#L91)
    * [x] hostname and IP cross-usage in ROS through the launch file. check [here](./src/rosprofiler/client.py#L44)
* [ ] PULL changes to the nano and compile

Check with client.py whether the changes are the same noted down in there
- [x] Units for the memory management need to be converted, see this [stackoverflow](https://stackoverflow.com/questions/21792655/psutil-virtual-memory-units-of-measurement)
- [x] Check the sample rates and intervals - do the values get reset each sample - yes, see [profiler.py, l. 176](./src/rosprofiler/profiler.py#L176)
- [x] Check the [psutil memory documentation](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info) if we can get more conclusive values. Also check [this stackoverflow](https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management) for what the differences between RSS, VMS and pss are
    * [ ] Which one do we use? Depends on ISO 25023 and 25010
        * [x] one to show the free space - for capacity
        * [x] for the host_statistics definitely want one that will show the total used space
        * [x] for the node_statistics want one that will show the allocated __and__ used space
- [x] The cpu_percent part of the [node_statistics, L. 56](./src/rosprofiler/node_monitor.py#L56) may be very misleading - check these and find if there is maybe an absolute value we can use - or if we have to rely on the host_statistics value to then calculate the total usage
- [x] Run a test example with the topic and BW timer 


- [x] Create a toy example with listener / talker
- [x] Extend classes:
    - [x] Bw and HZ from online documentation. Check issuelog
    - [x] Get power values consumption - **For the Raspberry: Use the maximum Power consumption (which is likely limited by the USB) and multiply it with the CPU load as a proxy - arguing that heat dissipation will be the major determining factor, with equal r/w access**
    - [x] Put power monitoring into message and send over
    - [x] Put different memory values into classes, s.a.
- [x] Adapt the code to use the new msg type monitoring power and new memory values
- [ ] Check with ISO what is required/ what we can and want to use
    * We want to look at the following values:
        * Performance
            * [ ] resource utilization
                * [x] Memory 
                    * [x] actually used memory
                        * [x] Per Process
                            * [x] RSS, **PSS** or USS, see the psutil [blogpost](http://grodola.blogspot.com/2016/02/psutil-4-real-process-memory-and-environ.html) - uss is slow
                            * [x] swap - may influence timing / longevity, etc.
                            * [x] virtual memory - how much is assigned
                        * [x] per Host
                            * [x] virtual memory - **used**, active see [documentation](https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory)
                            * [x] swap memory - used
                * [x] CPU usage
                    * [x] Per Process CPU_Percent is most relevant, can be bigger than 100 and can also be divided by num_cpus for windows-like behaviour, see [documentation](https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_percent)
                    * [x] Host cpu_percent - system-wide global utilization - not a list anymore - see [documentation](https://psutil.readthedocs.io/en/latest/#psutil.cpu_times)
                    * [x] Host logical CPU count - see [documentation](https://psutil.readthedocs.io/en/latest/#psutil.cpu_count)
                * [x] GPU
                    * [x] Omit usage, because we do not target the GPU (yet), however the extension would be trivial
                * [x] Timing - on ROS topics
                    * [x] Bandwith utilization
                    * [x] Frequency - hard limit
                    * [ ] The `get_topic_list()` function is pretty difficult to figure out. [Documentation on Format of rostopic.get_topic_list()](http://docs.ros.org/melodic/api/rostopic/html/rostopic-pysrc.html#get_topic_list)
                    * [x] use [this one](http://docs.ros.org/melodic/api/rospy/html/rospy.client-module.html#get_published_topics). is easier
            * [x] Capacity - how much can it access
                * [x] CPU - can be calculated from the cpu_count and the Host CPU percent
                * [x] Memory
                    * [x] Process
                        * [x] **NONE** process has VM available, which is also a used resource
                    * [x] Host
                        * [x] **MINIMUM** [available virtual memory](https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory) - conservative measure. This excludes swap
                        * [x] Swap in bytes 
                            * [x] available
                * [ ] Timing - on ROS topics - has to be done in postprocessing
                    * [ ] Bandwidth Limits - see documentation
                        * [ ] Wifi / Ethernet
                        * [ ] Serial, see below
                    * [ ] Frequency - hard limits
        * Compatibility
            * [x] Co-Existence - free memory
                * [x] Host Memory
                    * [x] Available Memory [documentation](https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory) - be careful here with overhead. Define what has been left out and what not - accessing PSS and USS is claimed to be expensive (leafing through memory paging)
            * [x] Interoperability - shared memory - **NOT** - left out
                * [x] Host Memory
                    * [x] Shared - [documentation](https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory)

- [x] Implement the option to change the frequency of monitoring.
    * [ ] [here](./scripts/rosprofilerNano#L52) using command line arguments 
    * [x] [here](./src/rosprofiler/profiler_Nano.py#L71) using ROS parameters

- [x] Implement the option to only monitor certain nodes:
    - [x] change [this](./src/rosprofiler/profiler_Nano.py#L94) to accept the rosparams set by the .config file in the `rosprofiler` package

- [x] Change the fields the client accepts and
     * [x] Host Statistics Nano, see [here](./src/client.py#L49) and [here](./src/client.py#L48)
     * [x] Node Statistics Nano, see [here](./src/client.py#L75) and [here](./src/client.py#L61)
     * [x] which nodes to log is double covered, see [here](./src/client.py#L) and [here](./src/rosprofiler/profiler_Nano.py#L142)
- [x] Launch file for starting timing logging together with client logging
    - [x] resolve namespace grouping issues, topic initialisation etc.
    * [x] Include Timing node 
- [x] Find a way to profile the rosprofiling node on the client
    - [x] cannot append to [rosparam list](./config/profileparams.yaml)
    - [ ] command line argument with [this syntax](./launch/nano_profiler.launch#L13)
    - [x] workaround inside the client, by looking for a combination of "nan" and "prof" in all the node name list? see [here](./src/rosprofiler/host_monitor_nano.py#L43) and [here](./src/client.py#L24) for reference

trial run to link to a line in a commit [here](./src/rosprofiler/profiler_Nano.py#L15@d05115ea722b46f9e84e259117c3ed09fc327460)

### Nico Result Processing TODO:

* [ ] Comparing the __same__ nodes (processes) on two different plattforms
    * [ ] Values to compare, see `NodeStatisticsNano msg`
        * [ ] absolute CPU hogging time - what is the CPU load defined by psutil
        * [x] PSS
        * [x] Virtual Memory
        * [x] Swap Memory
* [ ] Comparing the __host__ values (see `NanoStatistigs msg`) when increasing the number of nodes run on it - __same__ test case
    1. [ ] Worst case (Real-Time) - only necessary nodes: Image processing and waypoints
    2. [ ] Best case (Online) - full load
    * [x] Values to compare:
        * [x] Available Memory
        * [x] Maximum CPU load
        * [x] Power Draw - calculated either through CPU*nominal load or through monitoring values
        * [x] Faults
* [ ] Comparing the influence of relative weighting put into the AHP
    * [ ] When considering that Navigation is a supporting function - How does this change our values?
    * [ ] Under which Conditions are the following true:
       * [ ] Our conditions are consistent (see AHP CR)
       * [ ] The raspberry pi would be a better choice?

## Nico Potential Fixes

* [x] check if the `_update_node_list`- function, [here](./src/rosprofiler/profiler_Nano.py#L148) does not work due to mavros length!
    * [x] should not -since it is only one node, just running a fuckton of topics
* [x] Check what unit the Nano power logging is in - **MW** - see [here](https://forums.developer.nvidia.com/t/power-consumption-monitoring/73608/10)
* [x] Adding a file which will shutdown the entire process - see [here](./src/shutdown.py)

## NICO Nice-to-Haves

1. udp port usage, to see how much mavlink actually uses, check [this documentation](https://psutil.readthedocs.io/en/latest/#psutil.net_connections)
    * `For example 9600 8 N 1 uses 10 bits per word (1 start bit, 8 data bits, and 1 stop bit). Each word would take 10/9600 = 1041.66666666 microsecs`  to monitor whether we get channel overload on the udp port
    * use a switch in the launch file and put this into a separate file
    * Cannot work - due to socket locking, the traffic could only be monitored through other processes. disregard (for now)
2. Heat dissipation sensing, see [documentation](https://psutil.readthedocs.io/en/latest/#psutil.sensors_temperatures)

## Modifications made by Nico to the hardware setup

* Px4 Firmware:
    * [ ] Version: 1.10.1
    * [ ] Changed parameters 
    * [ ] mavros_extras have to be compiled
    * [ ] Hardware Power usage for consideration when selecting power distribution - see [this](https://diydrones.com/profiles/blogs/pixhawk-and-apm-power-consumption)
* Raspberry
    * [ ] cross-compiled OS ubuntu mate 18.04 - RAM issues
    * [x] enabled zram - Ram/cores /2 - roughly 256 MB
    * [x] added persistent 1 G swap - memory card longevity, swap paging
    * [x] disabled hdmi - for power savings
    * [x] disabled display service - for RAM savings
    * [x] disabled bluetooth - Device Tree to use the UART. Not enabled CTS and RTS
    * [x] custom opencv version - cross-compiled from source on B+ (overnight job)
    * [x] Serial clock speed elevated to work with mavros
    * [ ] Only works on WiFi - ROuter placed in direct LOS  @ less than 1m Distance - 5 Ghz prefered
    * [ ] Tmux
    * [ ] Mavros 0.33.4 - md5 checksum images - compilation times ~ 7 hrs (+9hrs for extras)
    * [ ] SSH key access - roslaunch - hostname parsing
    * [ ] compiled `image_transport_plugins` - for compressed image transport due to WiFi only (Compilation time ~ 4 hrs) - due to weird OpenCV dependencies
    * [ ] ntp time sync
    * [ ] assigning process priority, see link from [qutas wiki](https://github.com/qutas/info/wiki/General-Computing#changing-software-priority), also possible in ROS with a [launch-prefix](https://answers.ros.org/question/246090/how-to-launch-nodes-with-realtime-priority/)
    * [ ] Power issues, see this link from [qutas wiki](https://github.com/qutas/info/wiki/General-Computing#wifi-issues)
    * [ ] Git commit hash/release of used packages
    * [ ] IF there is no subscriber to a topic - there will be no message sent - TCP sockets. [If a tree falls in the forest, will it make a sound?](https://answers.ros.org/question/173813/why-publish-if-no-one-is-subscribing/) - extra topics should not elevate bandwidth
    * [ ] timesync rate dropped during WiFi connection - assumed because of mavros subscriber on host pc
        * Test by:
            * [ ] Running the node in question on the same pc
            * [ ] Running the system on the nano with the wifi adapter
* Nano
    * [ ] disabled display service - save ram
    * [ ] enabled high power mode - power supply, also peripherals, see [these considerations](https://forums.developer.nvidia.com/t/power-supply-considerations-for-jetson-nano-developer-kit/71637)
    * [ ] jetpack version does not support zram
    * [ ] custom opencv version - compiled from source
    * [ ] enabled swapfile - 2G - sd card longevity
    * [ ] connected on ethernet - image transport
    * [ ] tmux
    * [ ] mavros 0.33.4 - md5 checksum issues
    * [ ] SSH key access - roslaunch
    * [ ] compiled image_transport_plugins - for compressed image transport
    * [ ] ntp time sync
    * [ ] WiFi only on 2.4 GhZ - limited compatible modules available for Nano
        * [ ] Start once to sync up mavlink parameters

* Setting niceness values below 0 requires super user privileges 

## Result Processing

### 1 - running initial tests on Nano and Raspberry
Raspberry exhibits dropouts (0-performance indicators) under minimal load. To elimiate this being due to data transfer of raw images, we use compressed images and modify the transport mode for wifi. The issue persists, so we do the same test with the nano. This does not exhibit changes, so we test the full load with compressed images on the Nano.

### 2 - AHP
* [ ] CPU used
    * [ ] Not normally distributed - using the 90th percentile of maximum CPU load - robust and in line with hard performance models
    * [ ] Criteria
        * [ ] Higher is better
        * [ ] Piecewise subtraction
           * [ ] if  < 1: 1/, if > 1 -> difference scaled 

* [ ] Memory used
    * [ ] Used Memory Max. virt mem used - not the best indicator, but sufficient - contains underlying process
    * [x] Nominal values:
        * [x] Raspberry Pi - `512` MB
        * [x] Nano - `4096` MB
    * [ ] Criteria
        * [ ] % of memory used: relative value
            * [ ] piecewise subtraction
        * [ ] Higher is better

* [ ] CPU free
    * [ ] 1 - CPU-used. as a safety for system resources and cycling things
    * [ ] Criteria
        * [ ] % of usage - relative value
            * [ ] piecewise subtraction
        * [ ] Higher is better

* [ ] Memory free
    * [ ] Avail memory Min - virt mem avail - instantly given without going into swap - very accurate
    * [ ] Criteria
        * [ ] absolute value
            1. [ ] Take the maximum nominal value of all compared devices
            2. [ ] Take the % of THAT
        * [ ] %. Use piecewise subtraction
        * [ ] Higher is better

* [x] Faults
    * [x] Amount of times the mean CPU load fell under 0.1 - indicator of system dropout from preliminary experiments
    * [x] Criteria
        * [x] absolute value
        * [x] > 1 very bad. If all are > 1, the ratio can be used, but both are bad. 
        * [x] use: 0 vs x: 9 - (ln(max/x_i)
            and x vs 0: 1/ (9 - ln(max/x_i)
            and x_i vs x_j: 1+ ln(x_i/x_j). if result is smaller than 1, use 1/
            see functions [weirdstuff](./src/teststuff.py#L46) and [testotherstuff](./src/teststuff.py#L66)
* [ ] Size
    * [x] volume of the outermost square, including case, calculated by `width*height*depth` (when looking from the front) in cm
        * [x] Nano: `12 x 6.2 x 9`
        * [x] Raspberry: `7 x 1.6 x 6`
    * could theoretically be split further: height is very bad - centre of gravity - flat is better
    * [ ] Criteria
        * [ ] absolute value
        * [ ] lower is better
        * [ ] upper hard limit? What could be used here?

* [x] Weight of the Drone:
    * [x] Weight the drone itself (with battery), `1322 g`
    * [x] Add onboard computers + cases (+ wifi adapter). All other cables assumed equal
        * [x] Nano: `252 g`
        * [x] Raspberry: `44 g`
    * [ ] Criteria:
        * [ ] lower is better
        * [ ] if > 2.0 kg then 1/9

* [x] Power.
    * [x] Measured: jetson Nano - mean
    * [x] Calculated:  Average CPU load * nominal rating: Raspbi 4.1 W 
    * [x] Criteria:
       * [x] lower is better - no hard limit
       * [x] relative or absolute scaling?
           * ballpark figure: Drones run between 10 and 15 V, depending on battery size. Charge of a 4 Ah rated battery (lasts 1h @ 4Amp draw).
           Drones have a flying time of roughly 20minutes (excluding safety, depleting about 80% of charge), estimating to 3200 draw / 1/3 hour -> leads to 9.6 A, rounding 10A , with 14V -> roughly 140 Watts
           using the decrease in flight time due to power consumption. tf = (V*(IaRating * safety) / P0 + Pe) with Pe = V*I and P0 trough t0. Then we divide t0 - tf by t0 to get the relative decrease in flight time. 
           other calculation [here](https://www.quora.com/What-is-the-power-consumption-of-a-drone-like-the-Phantom-DJI)
        * [x] relative scaling - multiplication

### Bonus AHP stuff which could be used:
* [ ] Does the system go into swap
* [ ] The entire timing stuff, which we neglect for this
* [ ] Process-wise monitoring

### Conclude
We will __NOT__ take this on the drone. Research on comparing the HITL on the platforms has shown that the system is susceptible to errors, which require a fundamental redesign of the system.
We do not see this as a failure, but as a success for the evaluation step presented in this research, preventing premature deployment with potentially severe consequences. The stepwise migration and assessment of the system performance underlines the suggestions voiced by experienced researchers with quantitative data.
WiFi influence is documented [in this paper](https://www.sciencedirect.com/science/article/pii/S0921889017309144)

# Stuff from the original documentation


[![Build Status](https://travis-ci.org/osrf/rosprofiler.svg?branch=master)](https://travis-ci.org/osrf/rosprofiler)
[![Build Status](http://jenkins.ros.org/buildStatus/icon?job=devel-indigo-rosprofiler)](http://jenkins.ros.org/job/devel-indigo-rosprofiler/)

# UPDATES:

* Changed Naming conventions accordint to [this link](http://grodola.blogspot.com/2015/06/psutil-30.html)
* Changed l. 57 of node_monitor to fit to correct memory allocations, see this link on [memory_info()](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info)