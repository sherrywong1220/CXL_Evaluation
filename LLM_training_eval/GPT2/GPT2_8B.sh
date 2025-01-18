#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
bs=3

nl=72
h=3072
np=100
config_json="$script_dir/GPT2_8B_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads 32 \
       --batch-size ${bs} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${np} \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --log-interval 5 \
       --cpu-optimizer \
       &> gpt2_zero_2_numactl_all_layers_${nl}_hidden_${h}_bs_${bs}.log
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=all \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x


gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads 32 \
       --batch-size ${bs} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${np} \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --log-interval 5 \
       --cpu-optimizer \
       &> gpt2_zero_2_numactl_0,2_layers_${nl}_hidden_${h}_bs_${bs}.log
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=0,2 \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x


gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads 32 \
       --batch-size ${bs} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${np} \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --log-interval 5 \
       --cpu-optimizer \
       &> gpt2_zero_2_numactl_0,1_layers_${nl}_hidden_${h}_bs_${bs}.log
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --interleave=0,1 \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nl} \
       --hidden-size ${h} \
       --num-attention-heads 32 \
       --batch-size ${bs} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${np} \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --log-interval 5 \
       --cpu-optimizer \
       &> gpt2_zero_2_numactl_DDR0_layers_${nl}_hidden_${h}_bs_${bs}.log
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="numactl --membind=0 \
       deepspeed --num_nodes ${NUM_WORKERS} \
              --num_gpus ${NUM_GPUS_PER_WORKER} \
              pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x


