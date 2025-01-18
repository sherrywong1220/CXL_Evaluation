#!/bin/bash

BIN=/home/dongl/workloads/silo/out-perf.masstree/benchmarks
BENCH_RUN="${BIN}/dbtest --verbose --bench ycsb --num-threads 64 --scale-factor 900000 --ops-per-worker=200000000"

# Mem size: 130GB
