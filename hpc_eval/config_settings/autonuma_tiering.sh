#!/bin/bash
echo 2 | sudo tee /proc/sys/kernel/numa_balancing
echo 1 | sudo tee /sys/kernel/mm/numa/demotion_enabled