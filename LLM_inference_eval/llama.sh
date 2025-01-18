numactl --cpubind 1 --membind 1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 10 --gpu-batch-size 3 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 0 8 86 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 1,2  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 10 --gpu-batch-size 3 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 8 92 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 10 --gpu-batch-size 3 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 8 92 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --membind 1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 14 --gpu-batch-size 1 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 20 80 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 10 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 4 96 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1,2  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 14 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 4 96 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1,2  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-65b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 14 --gpu-batch-size 4 --sep-layer False --pin-weight False --percent 0 100 4 96 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

echo "now llama30b"

numactl --cpubind 1 --membind 1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 15 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 0 7 66 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 1,2  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 15 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 7 93 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 15 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 7 93 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --membind 1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 9 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 11 89 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1  python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 19 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 1 99 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1,2 python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 28 --gpu-batch-size 4 --cpu-cache-compute --sep-layer False --pin-weight False --percent 0 100 4 96 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"

numactl --cpubind 1 --interleave 0,1,2 python3 x-ai-main/x-ai/flexgen/flex_llama.py --model "huggyllama/llama-30b" --prompt-len 2048 --gen-len 256 --num-gpu-batches 28 --gpu-batch-size 4 --sep-layer False --pin-weight False --percent 0 100 4 96 0 100
rm -rf /mnt/nvme0n1-new-lv/data/flexgen_offload_dir
echo "clean"
