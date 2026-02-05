# CXL Evaluation
This is the repository that holds the artifacts of IPDPS'25 -- Performance Characterization of CXL Memory and Its Use Cases

## Prerequisite
`sudo apt-get install -y linux-tools-$(uname -r)`

`sudo apt-get install libnuma-dev`

## LLM Evaluation

### LLM Inference Evaluation
We use FlexGen framework to run large language models with a Single GPU.

Under `./LLM_inference_eval/`


#### Evaluate OPT

```
bash ./opt.sh
```

#### Evaluate LLaMA

```
bash ./llama.sh
```


### LLM Training Evaluation

Under `./LLM_training_eval/`

#### Evaluate BERT
evaluate BERT with different parameters sizes
```
bash BERT_base.sh
bash BERT_4B.sh
bash 
```

#### Evaluate GPT-2

evaluate GPT-2 with different parameters sizes
```
bash GPT2_4B.sh
bash GPT2_6b.sh
bash GPT2_8B.sh
```




## HPC Evaluation
Under `./hpc_eval/`

### Configurations
Set arguments using environment variables in `run_2_tiering.sh` and `run_3_tiering.sh`

| Argument | Brief description  | Valid inputs |
| -------- | ----------------- | ------- 
| BENCHMARKS | a list of benchmarks to test | XSBench PageRank Silo BTree Graph500 ...|
| MEM_POLICYS | a list of NUMA Policies | cpu1.membind1_2 cpu1.membind0_1 cpu1.interleave0_1 cpu1.interleave1_2 cpu01.firsttouch1_2 cpu01.interleave1_2 ...|
|OMP_NUM_THREADS| The number of threads | 32|
|VER|Memory tiering system |autonuma tiering-0.8 tpp memtis autonuma_tiering nobalance|


A configuration example
```
BENCHMARKS="XSBench PageRank Silo BTree Graph500 BT BT_interleave"
MEM_POLICYS="cpu1.firsttouch1_2 cpu1.interleave1_2"
export NPB_NUMA_NODES=1,2
export OMP_NUM_THREADS=32
# VER="autonuma"
# VER="tiering-0.8"
# VER="tpp"
# VER="memtis"
# VER="autonuma_tiering"
VER="nobalance"
```

### tests for applications running on two tiers
```
sudo ./run_2_tiering.sh
```

### tests for applications running on three tiers
```
sudo ./run_3_tiering.sh
```

## CXL Characterization Test

Thanks to the marvelous work in this [publication (MICRO23-Sun)](https://dl.acm.org/doi/10.1145/3613424.3614256) and their [repository](https://github.com/ece-fast-lab/cxl_type3_tests.git), which facilitated our CXL characterization tests.


Under `./cxl_characterization/test_cxl/`

### Latency test
```
bash test_ptr_chase.sh 
```

### `movdir64B` bandwidth
```
bash test_movdir_bw.sh
```

### Sequential access bandwidth 
```
bash test_seq_bw.sh
```

### Random access bandwidth
```
bash test_rand_bw.sh
```

### Loaded Latency
./mlc --loaded_latency
