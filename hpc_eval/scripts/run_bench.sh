#!/bin/bash

# Todo: change DIR
export DIR=/home/dongl/run_bench

function func_cache_flush() {
    echo 3 > /proc/sys/vm/drop_caches
    free
    return
}

function func_prepare() {
    echo "Preparing benchmark start..."

	ulimit -m unlimited
	ulimit -v unlimited
	ulimit -d unlimited
	ulimit -s unlimited

	sudo sysctl kernel.perf_event_max_sample_rate=100000
	# echo 1 > /sys/kernel/debug/tracing/events/migrate/mm_migrate_pages/enable


    if [[ -e ${DIR}/config_settings/${VER}.sh ]]; then
	    source ${DIR}/config_settings/${VER}.sh
	else
	    echo "ERROR: ${VER}.sh does not exist."
	    exit -1
	fi
	sudo killall -9 vmstat.sh
	
	DATE=$(date +%Y%m%d%H%M)

	if [[ -e ${DIR}/bench_cmds/${BENCH_NAME}.sh ]]; then
	    source ${DIR}/bench_cmds/${BENCH_NAME}.sh
	else
	    echo "ERROR: ${BENCH_NAME}.sh does not exist."
	    exit -1
	fi

    if [[ -e ${DIR}/mem_policies/${MEM_POLICY}.sh ]]; then
	    source ${DIR}/mem_policies/${MEM_POLICY}.sh
	else
	    echo "ERROR: ${MEM_POLICY}.sh does not exist."
	    exit -1
	fi
}

function func_usage() {
    echo
    echo -e "Usage: $0 [-B benchmark] [-V version] [-M mempolicy] [-LM localmem]..."
    echo
    echo "  -B,   --benchmark   [arg]    benchmark name to run. e.g., Graph500, XSBench, etc"
    echo "  -V,   --version     [arg]    version to run. e.g., autonuma, TPP, etc"
	echo "  -M,   --mempolicy   [arg]    memory policy to run. e.g., cpu1.membind0_1_2, cpu1.membind1_2, etc"
	echo "  -LM,  --localmem    [arg]    local memory size. e.g., 65G, 105G, etc"
    echo "        --cxl                  enable cxl mode [default: disabled]"
    echo "  -?,   --help"
    echo "        --usage"
    echo
}

# get options:
while (( "$#" )); do
    case "$1" in
	-B|--benchmark)
	    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
		BENCH_NAME=( "$2" )
		shift 2
	    else
		echo "Error: Argument for $1 is missing" >&2
		func_usage
		exit -1
	    fi
	    ;;
	-V|--version)
	    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
		VER=( "$2" )
		shift 2
	    else
		echo "Error: Argument for $1 is missing" >&2
		func_usage
		exit -1
	    fi
	    ;;
	-M|--mempolicy)
	    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
		MEM_POLICY=( "$2" )
		shift 2
	    else
		echo "Error: Argument for $1 is missing" >&2
		func_usage
		exit -1
	    fi
	    ;;
	-LM|--localmem)
	    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
		LOCAL_MEM=( "$2" )
		shift 2
	    else
		echo "Error: Argument for $1 is missing" >&2
		func_usage
		exit -1
	    fi
	    ;;
	-P|--perf)
	    CONFIG_PERF=on
	    shift 1
	    ;;
	-NS|--nosplit)
	    CONFIG_NS=on
	    shift 1
	    ;;
	-NW|--nowarm)
	    CONFIG_NW=on
	    shift 1
	    ;;
	--cxl)
	    CONFIG_CXL_MODE=on
	    shift 1
	    ;;
	-H|-?|-h|--help|--usage)
	    func_usage
	    exit
	    ;;
	*)
	    echo "Error: Invalid option $1"
	    func_usage
	    exit -1
	    ;;
    esac
done

function func_main() {
    TIME="/usr/bin/time"

    if [[ "x${CONFIG_PERF}" == "xon" ]]; then
	PERF="./perf stat -e dtlb_store_misses.walk_pending,dtlb_load_misses.walk_pending,dTLB-store-misses,cycle_activity.stalls_total"
    else
	PERF=""
    fi

    # make directory for results
    mkdir -p ${DIR}/results/${BENCH_NAME}/${VER}/${MEM_POLICY}/${LOCAL_MEM}
    LOG_DIR=${DIR}/results/${BENCH_NAME}/${VER}/${MEM_POLICY}/${LOCAL_MEM}

	cat /proc/vmstat | grep -e thp -e htmm -e migrate -e pgpromote -e pgdemote -e numa -e promote > ${LOG_DIR}/before_vmstat.log
    func_cache_flush
    sleep 2

    ${DIR}/scripts/vmstat.sh ${LOG_DIR} &
    if [[ "x${BENCH_NAME}" =~ "xsilo" ]]; then
	CMD="stdbuf -oL -eL ${TIME} -f 'execution time %e (s)' ${PINNING} ${BENCH_RUN} 2>&1 | tee ${LOG_DIR}/output.log"
	echo ${CMD}
	stdbuf -oL -eL ${TIME} -f "execution time %e (s)" \
	    ${PINNING} ${BENCH_RUN} 2>&1 \
	    | tee ${LOG_DIR}/output.log
    else
	CMD="stdbuf -oL -eL ${TIME} -f 'execution time %e (s)' ${PINNING} ${BENCH_RUN} 2>&1 | tee ${LOG_DIR}/output.log"
	echo ${CMD}
	stdbuf -oL -eL ${TIME} -f "execution time %e (s)" \
	    ${PINNING} ${BENCH_RUN} 2>&1 \
	    | tee ${LOG_DIR}/output.log
    fi

    sudo killall -9 vmstat.sh
	cat /proc/vmstat | grep -e thp -e htmm -e migrate -e pgpromote -e pgdemote -e numa -e promote > ${LOG_DIR}/after_vmstat.log
    sleep 2

    if [[ "x${BENCH_NAME}" == "xbtree" ]]; then
	cat ${LOG_DIR}/output.log | grep Throughput \
	    | awk ' NR%20==0 { print sum ; sum = 0 ; next} { sum+=$3 }' \
	    > ${LOG_DIR}/throughput.out
    elif [[ "x${BENCH_NAME}" =~ "xsilo" ]]; then
	cat ${LOG_DIR}/output.log | grep -e '0 throughput' -e '5 throughput' \
	    | awk ' { print $4 }' > ${LOG_DIR}/throughput.out
    fi

	# sudo cat /sys/kernel/debug/tracing/trace > ${LOG_DIR}/trace.txt

    sudo dmesg -c > ${LOG_DIR}/dmesg.txt
}

func_prepare
func_main
