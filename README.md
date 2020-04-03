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
- [ ] Units for the memory management need to be converted, see this [stackoverflow](https://stackoverflow.com/questions/21792655/psutil-virtual-memory-units-of-measurement)
- [x] Check the sample rates and intervals - do the values get reset each sample - yes, see [profiler.py, l. 176](./src/rosprofiler/profiler.py#L176)
- [ ] Check the [psutil memory documentation](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info) if we can get more conclusive values. Also check [this stackoverflow](https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management) for what the differences between RSS, VMS and pss are
    * [ ] Which one do we use? Depends on ISO 25023 and 25010
        * [ ] one to show the free space - for capacity
        * [ ] for the host_statistics definitely want one that will show the total used space
        * [ ] for the node_statistics want one that will show the allocated __and__ used space
- [ ] The cpu_percent part of the [node_statistics, L. 56](./src/rosprofiler/node_monitor.py#L56) may be very misleading - check these and find if there is maybe an absolute value we can use - or if we have to rely on the host_statistics value to then calculate the total usage


- [x] Create a toy example with listener / talker
- [x] Extend classes:
    - [x] Bw and HZ from online documentation. Check issuelog
    - [x] Get power values consumption - **For the Raspberry: Use the maximum Power consumption (which is likely limited by the USB) and multiply it with the CPU load as a proxy - arguing that heat dissipation will be the major determining factor, with equal r/w access**
    - [ ] Put power monitoring into message and send over
    - [ ] Put different memory values into classes, s.a.
- [ ] Adapt the code to use the new msg type monitoring power and new memory values
- [ ] Check with ISO what is required/ what we can and want to use


# Stuff from the original documentation


[![Build Status](https://travis-ci.org/osrf/rosprofiler.svg?branch=master)](https://travis-ci.org/osrf/rosprofiler)
[![Build Status](http://jenkins.ros.org/buildStatus/icon?job=devel-indigo-rosprofiler)](http://jenkins.ros.org/job/devel-indigo-rosprofiler/)

# UPDATES:

* Changed Naming conventions accordint to [this link](http://grodola.blogspot.com/2015/06/psutil-30.html)
* Changed l. 57 of node_monitor to fit to correct memory allocations, see this link on [memory_info()](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info)