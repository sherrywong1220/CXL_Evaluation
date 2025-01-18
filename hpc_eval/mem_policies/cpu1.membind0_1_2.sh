#!/bin/bash

# PINNING="numactl -N 1 -m 0,1,2 --"
PINNING="numactl -C 32-63 -m 0,1,2 --"