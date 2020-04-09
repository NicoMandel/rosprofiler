rosprofiler
===========

# Updates by Nico
* Changed [node_monitor.py](./src/rosprofiler/node_monitor.py) according to the psutil documentation
    * included correct named tuple referencing in the `update` function - mem_info

* Added a [client.py file](./src/client.py)
    * base structure is present
    * Depends on parameters to be set elsewhere which say what to log, otherwise will log everything by default, see [here](./src/client.py#L39)


## Nico TODOs:
Check with client.py whether the changes are the same noted down in there
- [x] Units for the memory management need to be converted, see this [stackoverflow](https://stackoverflow.com/questions/21792655/psutil-virtual-memory-units-of-measurement)
- [x] Check the sample rates and intervals - do the values get reset each sample - yes, see [profiler.py, l. 176](./src/rosprofiler/profiler.py#L176)
- [ ] Check the [psutil memory documentation](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info) if we can get more conclusive values. Also check [this stackoverflow](https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management) for what the differences between RSS, VMS and pss are
    * [ ] Which one do we use? Depends on ISO 25023 and 25010
        * [ ] one to show the free space - for capacity
        * [ ] for the host_statistics definitely want one that will show the total used space
        * [ ] for the node_statistics want one that will show the allocated __and__ used space
- [x] The cpu_percent part of the [node_statistics, L. 56](./src/rosprofiler/node_monitor.py#L56) may be very misleading - check these and find if there is maybe an absolute value we can use - or if we have to rely on the host_statistics value to then calculate the total usage
- [ ] Run a test example with the topic and BW timer 


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
                * [ ] Timing - on ROS topics
                    * [ ] Bandwith utilization
                    * [ ] Frequency - hard limit
            * [x] Capacity - how much can it access
                * [x] CPU - can be calculated from the cpu_count and the Host CPU percent
                * [x] Memory
                    * [x] Process
                        * [x] **NONE** process has VM available, which is also a used resource
                    * [x] Host
                        * [x] **MINIMUM** [available virtual memory](https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory) - conservative measure. This excludes swap
                        * [x] Swap in bytes 
                            * [x] available
                * [ ] Timing - on ROS topics
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

- [ ] Implement the option to change the frequency of monitoring.
    * [ ] [here](./scripts/rosprofilerNano#L52) using command line arguments 
    * [x] [here](./src/rosprofiler/profiler_Nano.py#L71) using ROS parameters
    * [ ] for the [timing](./src/timing.py#L13) Script

- [x] Implement the option to only monitor certain nodes:
    - [ ] change [this](./src/rosprofiler/profiler_Nano.py#L135) to be a function
    - [x] change [this](./src/rosprofiler/profiler_Nano.py#L94) to accept the rosparams set by the .config file in the `beginner_tutorials` package

- [ ] Change the fields the client accepts and
     * [ ] Host Statistics Nano, see [here]
     * [ ] Node Statistics Nano, see [here]
- [ ] Launch file for starting timing logging together with client logging


## NICO Nice-to-Haves

1. udp port usage, to see how much mavlink actually uses, check [this documentation](https://psutil.readthedocs.io/en/latest/#psutil.net_connections)
    * ` For example 9600 8 N 1 uses 10 bits per word (1 start bit, 8 data bits, and 1 stop bit). Each word would take 10/9600 = 1041.66666666 microsecs.` - to monitor whether we get channel overload on the udp port
2. Heat dissipation sensing, see [documentation](https://psutil.readthedocs.io/en/latest/#psutil.sensors_temperatures)

# Stuff from the original documentation


[![Build Status](https://travis-ci.org/osrf/rosprofiler.svg?branch=master)](https://travis-ci.org/osrf/rosprofiler)
[![Build Status](http://jenkins.ros.org/buildStatus/icon?job=devel-indigo-rosprofiler)](http://jenkins.ros.org/job/devel-indigo-rosprofiler/)

# UPDATES:

* Changed Naming conventions accordint to [this link](http://grodola.blogspot.com/2015/06/psutil-30.html)
* Changed l. 57 of node_monitor to fit to correct memory allocations, see this link on [memory_info()](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info)