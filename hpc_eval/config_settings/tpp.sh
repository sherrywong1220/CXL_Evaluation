#!/bin/bash
echo 1 | sudo tee /sys/kernel/mm/numa/demotion_enabled
echo 2 | sudo tee /proc/sys/kernel/numa_balancing
echo 200 | sudo tee /proc/sys/vm/demote_scale_factor


# Inherit AutoNUMA configs
echo 256 | sudo tee /sys/kernel/debug/sched/numa_balancing/scan_size_mb
echo 1000 | sudo tee /sys/kernel/debug/sched/numa_balancing/scan_delay_ms
echo 60000 | sudo tee /sys/kernel/debug/sched/numa_balancing/scan_period_max_ms
echo 1000 | sudo tee /sys/kernel/debug/sched/numa_balancing/scan_period_min_ms