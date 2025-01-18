#!/bin/bash
#### default setting


echo 2 | sudo tee /proc/sys/kernel/numa_balancing
echo 1 | sudo tee /sys/kernel/mm/numa/demotion_enabled

## enable numa balancing for promotion
echo 30 | sudo tee /proc/sys/kernel/numa_balancing_rate_limit_mbps

## enable early wakeup
echo 1 | sudo tee /proc/sys/kernel/numa_balancing_wake_up_kswapd_early

## enable decreasing hot threshold if the pages just demoted are hot
echo 1 | sudo tee /proc/sys/kernel/numa_balancing_scan_demoted
echo 16 | sudo tee /proc/sys/kernel/numa_balancing_demoted_threshold