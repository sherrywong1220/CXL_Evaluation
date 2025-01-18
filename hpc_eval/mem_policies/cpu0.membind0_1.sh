#!/bin/bash

# PINNING="numactl -N 0 -m 0,1 --"
PINNING="numactl -C 0-31 -m 0,1 --"