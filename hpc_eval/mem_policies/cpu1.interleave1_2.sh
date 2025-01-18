#!/bin/bash

PINNING="numactl -N 1 --interleave=1,2 --"
# PINNING="numactl -C 32-63 --interleave=1,2 --"