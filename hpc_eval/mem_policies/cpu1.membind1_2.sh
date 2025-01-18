#!/bin/bash

# PINNING="numactl -N 1 -m 1,2 --"
PINNING="numactl -C 32-63 -m 1,2 --"