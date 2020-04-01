rosprofiler
===========

# Updates by Nico
* Changed [node_monitor.py](./src/rosprofiler/node_monitor.py) according to the psutil documentation
    * included correct named tuple referencing in the `update` function - mem_info

* Added a [client.py file](./src/client.py)
    * base structure is present


## Nico TODOs:
Check with client.py whether the changes are the same noted down in there
* Units for the memory management need to be converted, see this [stackoverflow](https://stackoverflow.com/questions/21792655/psutil-virtual-memory-units-of-measurement)
* Check the sample rates and intervals - do the values get reset each sample?
* Check the [psutil memory documentation](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info) if we can get more conclusive values. Also check [this stackoverflow](https://stackoverflow.com/questions/7880784/what-is-rss-and-vsz-in-linux-memory-management) for what the differences between RSS, VMS and pss are 


* Create a toy example with listener / talker
* Extend classes:
    * Bw and HZ from online documentation. Check issuelog
    * Get power values consumption
* Check with ISO what is required


# Stuff from the original documentation


[![Build Status](https://travis-ci.org/osrf/rosprofiler.svg?branch=master)](https://travis-ci.org/osrf/rosprofiler)
[![Build Status](http://jenkins.ros.org/buildStatus/icon?job=devel-indigo-rosprofiler)](http://jenkins.ros.org/job/devel-indigo-rosprofiler/)

# UPDATES:

* Changed Naming conventions accordint to [this link](http://grodola.blogspot.com/2015/06/psutil-30.html)
* Changed l. 57 of node_monitor to fit to correct memory allocations, see this link on [memory_info()](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info)