#!/bin/bash

BENCHMARKS="XSBench PageRank Silo BTree Graph500 BT BT_interleave"
BENCHMARKS="LU SP LU_interleave SP_interleave"
BENCHMARKS="MG MG_interleave"
# BENCHMARKS="XSBench_interleave"
# BENCHMARKS="LU_interleave SP_interleave"
# MEM_POLICYS="cpu1.membind1_2 cpu1.membind0_1 cpu1.interleave0_1 cpu1.interleave1_2"
# need change BENCH_RUN="NPB_NUMA_NODES=0,1 ${BIN}/bt.E"
# need change BENCH_RUN="NPB_NUMA_NODES=1,2 ${BIN}/bt.E"
# MEM_POLICYS="cpu01.firsttouch1_2 cpu01.interleave1_2"
MEM_POLICYS="cpu1.firsttouch1_2 cpu1.interleave1_2"
MEM_POLICYS="cpu1.firsttouch1_2"
export NPB_NUMA_NODES=1,2
# export NPB_NUMA_NODES=0,1

export OMP_NUM_THREADS=32

# sudo dmesg -c

# VER="autonuma"
# VER="tiering-0.8"
# VER="tpp"
# VER="memtis"
# VER="autonuma_tiering"
VER="nobalance"
# LOCAL_DRAM_SIZE="40G"
# LOCAL_DRAM_SIZE="65G"
#LOCAL_DRAM_SIZE="100G"
#LOCAL_DRAM_SIZE="120G"

LOCAL_DRAM_SIZE="50G"

mkdir -p log

for BENCH in ${BENCHMARKS};
do
    # if [[ "x${BENCH}" =~ "interleave" ]]; then
    # MEM_POLICYS="cpu1.firsttouch1_2"
    # else
    # MEM_POLICYS="cpu1.firsttouch1_2 cpu1.interleave1_2"
    # fi

    # if [[ "x${BENCH}" =~ "interleave" ]]; then
    # MEM_POLICYS="cpu01.firsttouch1_2"
    # else
    # MEM_POLICYS="cpu01.firsttouch1_2 cpu01.interleave1_2"
    # fi

    for MP in ${MEM_POLICYS};
    do
	# ./scripts/run_bench.sh -B ${BENCH} -M ${MP} -V ${VER} -LM ${LOCAL_DRAM_SIZE} >> ./log/${VER}_${CURRENT_TIME}.log

    CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "current_time ${CURRENT_TIME}"
    ./scripts/run_bench.sh -B ${BENCH} -M ${MP} -V ${VER} -LM ${LOCAL_DRAM_SIZE}
    sleep 15
    done
done
