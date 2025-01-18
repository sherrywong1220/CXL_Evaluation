#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/bert_4B_config.json"
bs=2
nd=40
nl=48
h=2560
np=1000

bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads ${nd} \
       --batch-size ${bs} \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters ${np} \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       &> bert_zero_2_numactl_DDR0_bs_${bs}_np_${np}.log
"
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --membind=0 \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_bert_cpu_adam.py $@ ${bert_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x

bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads ${nd} \
       --batch-size ${bs} \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters ${np} \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
        &> bert_zero_2_numactl_0,1_bs_${bs}_np_${np}.log
 "
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=0,1 \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_bert_cpu_adam.py $@ ${bert_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x

bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads ${nd} \
       --batch-size ${bs} \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters ${np} \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
        &> bert_zero_2_numactl_0,2_bs_${bs}_np_${np}.log
 "
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=0,2 \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_bert_cpu_adam.py $@ ${bert_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x

bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads ${nd} \
       --batch-size ${bs} \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters ${np} \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
        &> bert_zero_2_numactl_all_bs_${bs}_np_${np}.log
 "
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=all \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_bert_cpu_adam.py $@ ${bert_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
