numactl --cpubind 1 --interleave 0,1  python3 x-ai-main/x-ai/flexgen/flex_opt.py --model "facebook/opt-66b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 10 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 4 96 0 100

numactl --cpubind 1 --interleave 0,1  python3 x-ai-main/x-ai/flexgen/flex_opt.py --model "facebook/opt-66b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 10 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 6 94 0 100

numactl --cpubind 1 --interleave 0,1,2  python3 x-ai-main/x-ai/flexgen/flex_opt.py --model "facebook/opt-66b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 16 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 4 96 0 100

numactl --cpubind 1 --membind 1  python3 x-ai-main/x-ai/flexgen/flex_opt.py --model "facebook/opt-66b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 7 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 16 8 92 0 100

rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir

numactl --cpubind 1 --membind 1  python3 x-ai-main/x-ai/flexgen/flex_opt.py --model "facebook/opt-66b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 7 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 16 8 92 0 100

rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
