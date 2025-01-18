#!/bin/bash

# PINNING="numactl -N 1 --interleave=0,1,2 --"
PINNING="numactl -C 32-63 --interleave=0,1,2 --"