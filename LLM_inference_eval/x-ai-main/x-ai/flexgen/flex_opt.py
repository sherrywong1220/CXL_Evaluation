"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed
        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)


class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            # top_p=None
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        # print("self.policy.cpu_cache_compute: ", self.policy.cpu_cache_compute)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
#     prompts = ["""
# In a not-too-distant future, a global pandemic known as the Nexus Virus has swept across the world, leaving devastating consequences in its wake. Write a detailed account of a young doctor's journey as they navigate the chaotic healthcare system, grapple with ethical dilemmas, and witness the resilience of the human spirit in the face of adversity.

# Dr. Emily Johnson stood in the bustling hospital corridor, the cacophony of alarms and hurried footsteps filling the air. She adjusted her mask and glanced at her watch, realizing she had been on her feet for hours. The Nexus Virus had struck with an unprecedented ferocity, overwhelming healthcare systems worldwide.

# As a promising young doctor, Emily had dedicated her life to saving lives, but nothing could have prepared her for the relentless onslaught of patients. The virus had become a ruthless enemy, attacking indiscriminately, leaving countless victims in its wake. The hospital wards overflowed with the sick, their moans echoing through the halls.

# Each day, Emily confronted the ethical quandaries that had become alarmingly common. Supplies were scarce, and she found herself making heart-wrenching decisions on who received life-saving treatments. The weight of those choices bore heavily on her shoulders, and she wondered if she could ever find solace in the lives she saved, knowing there were countless others she couldn't.

# Amidst the chaos, Emily witnessed both the best and worst of humanity. She encountered selfless nurses who worked tirelessly, risking their own well-being to provide care. She also encountered those who exploited the crisis for personal gain, hoarding supplies or spreading misinformation. It was a constant battle against fear and misinformation, as people sought answers in a world of uncertainty.

# The toll on Emily's own mental and emotional well-being was undeniable. She saw the toll it took on her colleagues as well, as they grappled with the overwhelming loss and exhaustion. Yet, amidst the darkness, she found glimmers of hope. Patients who defied the odds, survivors who emerged from the grip of the virus, and communities that came together to support one another.

# Emily's commitment to her oath never wavered. She continued to fight, despite the exhaustion, the heartbreak, and the personal sacrifices. She worked tirelessly to provide care and comfort to those in need, knowing that her presence and compassion could make a difference, even in the face of overwhelming odds.

# As the months passed, the world slowly began to heal. New treatments and vaccines emerged, offering a glimmer of hope for a brighter future. Emily saw the devastating impact of the pandemic, but she also witnessed the resilience of the human spirit.

# In the end, the Nexus Virus would leave an indelible mark on humanity. It would reshape the healthcare systems, the way we interacted with one another, and the way we valued human life. It would serve as a stark reminder of our vulnerability, but also our capacity for strength and compassion.

# Dr. Emily Johnson's journey through the pandemic was just one of many, but it was a testament to the unwavering commitment of healthcare professionals and the power of empathy in the face of adversity. As the world emerged from the darkness, the lessons learned would shape a new era of preparedness, unity, and resilience, ensuring that no matter what challenges lay ahead, humanity would always rise to meet them.
# """]
    prompts = ["""Summarize the following story for me:

Part 1: The Enchanted Prelude

In the mystical land of Eldoria, where ancient magic and untold wonders danced with the gentle breeze, a tale unfolded. At the heart of Eldoria stood the city of Lumaria, a beacon of shimmering crystal spires that touched the sky.

Amidst Lumaria's bustling streets, there lived a young artisan named Seraphina. Known for her ability to sculpt dreams into tangible forms, Seraphina spent her days crafting intricate sculptures that seemed to breathe with a life of their own. Her art became a source of inspiration for the citizens of Lumaria, who believed that Seraphina possessed a rare connection to the enchanted realms.

One fateful night, as the twin moons of Eldoria cast their silvery glow upon Lumaria, a mysterious figure appeared at Seraphina's doorstep. Veiled in a cloak of starlight, the stranger introduced themselves as Astral, a celestial being from a distant constellation.

Astral spoke of a cosmic imbalance threatening Eldoria. The magical threads that wove the fabric of reality were unraveling, and Lumaria's crystal spires flickered with a disquieting resonance. Seraphina, chosen by the cosmic forces, was tasked with restoring harmony to Eldoria.

Guided by Astral, Seraphina embarked on a journey across the enchanted landscapes of Eldoria. Along the way, they encountered mythical creatures, ancient guardians, and hidden realms that existed beyond the veil of mortal perception. Each step brought them closer to the heart of the cosmic imbalance.

As they delved deeper into the mysteries of Eldoria, Seraphina and Astral uncovered a forgotten prophecy that foretold of a chosen artisan who would wield the power of dreams to mend the celestial tapestry. Seraphina's sculptures, imbued with the essence of the enchanted realms, held the key to restoring balance.

Yet, dark forces lurked in the shadows, eager to exploit the cosmic imbalance for their own malevolent purposes. A rival artisan, envious of Seraphina's connection to the celestial, sought to seize the unraveled threads and wield their power for personal gain.

The stage was set for a cosmic odyssey, where the fate of Eldoria hung in the delicate balance between light and shadow. As Seraphina and Astral ventured deeper into the unknown, the true magnitude of their quest began to unfold—a quest that would test the limits of their courage, creativity, and the enduring magic that bound them together.

Part 2: Echoes of the Celestial Melody

Seraphina and Astral journeyed through the mystic landscapes of Eldoria, their quest unfolding like a cosmic tapestry. The air crackled with latent magic as they navigated ancient forests, crossed shimmering rivers, and ascended crystalline peaks.

As the duo ventured deeper, the celestial imbalance manifested in surreal ways. Constellations flickered erratically, and Eldoria's once-steady magic surged and waned. Seraphina's sculptures, enchanted by Astral's celestial touch, resonated with the pulse of the unraveling threads, offering glimpses into the cosmic discord.

In the city of Eldor, renowned for its sacred libraries, Seraphina and Astral sought the guidance of the wise Oracle of Lumina. The Oracle, a venerable figure draped in robes adorned with celestial symbols, spoke in riddles that echoed through the ancient chambers.

"The celestial dance falters, threads fraying in the cosmic loom. Seek the Echo Stones," the Oracle intoned, her eyes reflecting the light of distant galaxies.

The Echo Stones, a mythical artifact hidden in the Whispering Caverns, held the power to amplify the celestial melody. With newfound purpose, Seraphina and Astral set forth, navigating the labyrinthine passages of the caverns, where whispers echoed like ancient hymns.

As they approached the heart of the Whispering Caverns, they encountered spectral guardians—ethereal beings born from the harmonies of the celestial realm. The guardians, recognizing Seraphina's connection to the enchanted realms, tested her resolve with trials that merged reality and dreams.

After overcoming the challenges, the duo reached the chamber where the Echo Stones lay ensconced in a bed of luminescent crystals. As Seraphina touched the stones, a surge of celestial energy coursed through her, revealing visions of Eldoria's inception and the delicate balance that held the universe in equilibrium.

Armed with the amplified celestial melody, Seraphina and Astral returned to Lumaria. The city, once veiled in disquiet, now resonated with renewed vibrancy. The crystal spires gleamed with a harmonious glow, reflecting the celestial balance that Seraphina had woven into the very fabric of Eldoria.

However, the rival artisan, fueled by jealousy and the allure of unchecked power, hatched a sinister plan. Unbeknownst to Seraphina and Astral, dark forces converged, threatening to unravel their hard-won harmony.

As the cosmic drama unfolded, the destiny of Eldoria teetered on the edge. The echoes of the celestial melody reverberated, entwining the fates of the mortal and celestial realms in an intricate dance—a dance that would determine whether Eldoria would embrace a future bathed in the radiant glow of cosmic unity or succumb to the shadows of discord.

Part 3: Shadows in the Celestial Dance

The cosmic dance continued, its rhythm echoing through the tapestry of Eldoria. Seraphina and Astral, unaware of the gathering shadows, reveled in the newfound harmony they had woven into the city of Lumaria.

Unbeknownst to them, the rival artisan, whose envy had festered into a consuming darkness, sought forbidden knowledge from the ancient texts within the Shadow Archives. In the depths of the archives, where forgotten whispers and malevolent energies intertwined, the artisan made a pact with a shadowy entity—a being hungry for the fragments of the unraveling celestial threads.

As the cosmic tapestry quivered, Lumaria began to experience strange disturbances. Unpredictable bursts of magic disrupted the once-steady enchantments, causing rifts between dimensions. The citizens, initially elated by the return of balance, now felt an undercurrent of unease.

Astral, attuned to the celestial pulse, sensed the growing discord. Together with Seraphina, they delved into the heart of Lumaria, seeking the source of the disturbances. Their journey led them to the Astral Observatory, a sacred place where the very fabric of the cosmos could be observed.

In the observatory, Astral and Seraphina gazed into the cosmic energies, their eyes reflecting the constellations that adorned the night sky. The vision revealed the dark pact made by the rival artisan, and the looming threat that shadowy tendrils posed to Eldoria's delicate balance.

To counter the growing shadows, Seraphina and Astral needed to harness the pure essence of the Enchanted Wells, mystical springs scattered across Eldoria that held the primordial magic of creation. The journey to the Wells became a race against time, as the shadows gained strength, threatening to unravel the celestial harmony.

As they ventured into the untamed realms, overcoming treacherous landscapes and confronting the ancient guardians of the Wells, Seraphina and Astral uncovered dormant powers within themselves. The very essence of Eldoria responded to their quest, and the Wells resonated with a luminous brilliance as the duo collected the lifeblood of creation.

Armed with the pure magic of the Enchanted Wells, Seraphina and Astral returned to Lumaria. The city, now shrouded in flickering shadows, awaited the final confrontation between light and darkness. The rival artisan, wielding the stolen celestial fragments, emerged from the depths of envy, ready to challenge the harmony Seraphina and Astral had worked so hard to restore.

The cosmic dance reached its crescendo as Seraphina and Astral faced the rival artisan in the heart of Lumaria, their every step echoing in the celestial tapestry. The fate of Eldoria hung in the balance, and the shadows cast long, threatening to engulf the once-illuminated city in an eternal night.
"""]

# Part 4: The Celestial Convergence

# In the heart of Lumaria, the cosmic confrontation unfolded. Seraphina and Astral stood against the rival artisan, whose once-enviable talent now twisted into a malevolent force fueled by stolen celestial fragments. Shadows danced around them, threatening to drown Lumaria in an eternal abyss.

# As the rival artisan unleashed waves of distorted magic, Seraphina and Astral weaved a counter-harmony using the pure essence of the Enchanted Wells. The celestial energies clashed, creating an ethereal symphony that reverberated through the crystal spires of Lumaria.

# The cosmic battle raged on, each clash of magic sending shockwaves across the enchanted city. Lumaria's fate teetered on the brink, the delicate threads of the celestial tapestry unraveling with each passing moment.

# Astral, drawing on the depths of celestial wisdom, devised a plan to break the rival artisan's hold on the stolen fragments. Seraphina, attuned to the enchanted realms, channeled her artistic prowess into a masterpiece—a sculpture that mirrored the cosmic dance and resonated with the celestial fragments' original melody.

# As the rival artisan's power reached its zenith, Seraphina unveiled her celestial sculpture. The harmonious energy it emitted pierced through the shadows, exposing the stolen fragments to the pure essence of the Enchanted Wells. The rival artisan, caught in the celestial convergence, witnessed the consequences of their misguided quest for power.

# In a burst of cosmic light, the stolen fragments returned to their rightful place, mending the unraveling threads of the celestial tapestry. Lumaria, bathed in the radiant glow, stood on the precipice of a harmonious rebirth.

# The rival artisan, humbled by the consequences of their actions, was enveloped in the celestial light. From the shadows emerged a transformed figure—a guardian of cosmic balance, tasked with protecting Eldoria from the allure of unchecked power.

# As the cosmic symphony settled, Lumaria blossomed into a city of unparalleled enchantment. The crystal spires pulsed with celestial radiance, and the citizens, once divided by uncertainty, now embraced the unity woven by Seraphina and Astral.

# Astral, having fulfilled the celestial mission, bid farewell to Seraphina, promising to watch over Eldoria from the distant reaches of the cosmos. Seraphina, her heart forever touched by the cosmic odyssey, continued her artistic journey, creating sculptures that mirrored the celestial harmony she had restored.

# In the wake of the cosmic conflict, Eldoria stood as a testament to the enduring magic of unity. The enchanted realms and mortal dimensions coexisted in a delicate balance, their destinies intertwined by the threads of the cosmic tapestry.

# And so, in the heart of Lumaria, where starlight and crystal spires met, the tale of Seraphina and Astral became a legend—a celestial saga echoing through the ages, reminding future generations of the transformative power of harmony, creativity, and the enduring magic that bound Eldoria together.

# Part 5: Echoes Across Time

# The cosmic symphony that had played out in Lumaria left an indelible mark on Eldoria. The city, once veiled in uncertainty, now thrived as a beacon of celestial harmony. Seraphina's sculptures, imbued with the essence of her cosmic journey, continued to inspire and uplift the citizens.

# As time passed, Seraphina's artistic talent evolved, and her creations reached new heights of intricacy and beauty. Her sculptures mirrored the ever-changing constellations in the Eldorian sky, capturing the essence of the celestial dance that had saved their world.

# Eldoria's prosperity rippled across the enchanted realms, drawing beings from distant dimensions who sought to witness the harmonious city. Scholars and dreamers alike gathered in Lumaria, contributing their knowledge and creativity to the ever-expanding tapestry of Eldoria.

# Astral, from the distant reaches of the cosmos, observed the city's transformation with pride. His celestial wisdom continued to guide Seraphina, and their connection remained unbroken, transcending the boundaries of space and time.

# As Eldoria thrived, the cosmic balance that Seraphina and Astral had restored became a timeless legend, recounted in the whispered tales of generations. The city's crystal spires, which had once flickered with disquiet, now shimmered with the enduring magic that bound their world together.

# But the echoes of their celestial journey did not end within Eldoria's borders. Beyond the city's walls, in the farthest corners of the enchanted realms, Seraphina's sculptures found their way to those in need of inspiration and hope.

# In distant realms, where magic intertwined with the mundane, individuals discovered Seraphina's art, and their lives were forever transformed. Each sculpture held a piece of the cosmic harmony, offering solace, guidance, and a reminder of the boundless potential that existed within every soul.

# As Eldoria's legacy echoed across time and dimensions, the celestial balance remained unshaken. The cosmic dance continued, weaving the destinies of all who encountered the radiant glow of Seraphina's art.

# And so, in the realms of Eldoria and beyond, the story of Seraphina and Astral became a timeless legend—a tale of courage, creativity, and the enduring magic of unity that resonated in the hearts of dreamers, artists, and seekers of celestial harmony.

# Part 6: The Celestial Convergence Unveiled

# Eldoria's legacy, guided by the cosmic harmony woven by Seraphina and Astral, continued to flourish and evolve over the centuries. Lumaria remained a beacon of celestial beauty and unity, drawing scholars, artists, and explorers from far and wide.

# Generations of Eldorians embraced the unique connection between their realm and the enchanted dimensions. The city's crystal spires glistened with the reflections of the myriad constellations that adorned the Eldorian sky. Seraphina's sculptures, scattered throughout Lumaria, continued to inspire awe and wonder.

# Yet, as time flowed onward, whispers of a celestial convergence grew among the Eldorians. It was said that once in a millennium, a rare celestial alignment would occur, revealing a hidden gateway to the heart of the cosmos itself.

# Seraphina, now a revered figure in Eldoria, felt the call of destiny. Guided by the knowledge imparted by Astral and her own connection to the enchanted realms, she embarked on a new quest—a quest to unlock the secrets of the celestial convergence.

# The celestial alignment drew closer, casting its radiant glow upon Eldoria. Seraphina delved into ancient texts, communed with Eldoria's most venerable scholars, and consulted her own celestial sculptures for guidance. The mysteries of the convergence unfolded like a cosmic puzzle.

# With the convergence imminent, Seraphina sought to unite the realms like never before. Eldorians and beings from the enchanted dimensions joined forces to prepare for the event. Magical wards were erected to protect Lumaria, and enchanted bridges were woven to connect Eldoria with the realms beyond.

# As the celestial alignment reached its zenith, the city of Lumaria transformed into a cosmic nexus. The sky shimmered with celestial colors, and the crystal spires resonated with the celestial convergence's energy. A gateway, pulsating with the essence of the cosmos, appeared at the heart of the city.

# With a deep breath and a heart full of determination, Seraphina stepped through the gateway, accompanied by a group of Eldorians and beings from the enchanted dimensions. They ventured into the cosmic realm, a place where the boundaries of reality and dreams intertwined.

# In this ethereal dimension, Seraphina and her companions encountered celestial beings, each representing a facet of the cosmic tapestry. Together, they embarked on a journey to restore balance and unlock the true potential of Eldoria's connection to the enchanted realms.

# The celestial convergence, once shrouded in mystery, now revealed its purpose—a convergence of hearts, minds, and realms. Seraphina's vision of unity had transcended time and dimensions, culminating in a celestial convergence that would forever bind the destinies of Eldoria and the enchanted realms.

# As they ventured deeper into the cosmic realm, the secrets of the convergence began to unfold, and the destiny of Eldoria's cosmic journey reached a pivotal moment—a moment that held the power to shape the future of their interconnected worlds.
# Part 7: The Cosmic Tapestry Revealed

# In the cosmic realm, Seraphina and her companions embarked on a journey of discovery, guided by the celestial beings they had encountered. The path they followed led them through surreal landscapes that reflected the essence of the enchanted realms.

# As they progressed, the celestial convergence unveiled its true purpose. It was not merely a convergence of celestial energies but a revelation of the interconnectedness of all existence. The cosmic realm, once shrouded in mystery, now became a canvas upon which the destiny of Eldoria and the enchanted dimensions unfolded.

# Seraphina's connection to the enchanted realms and her artistic prowess had been instrumental in guiding them to this cosmic nexus. Each companion possessed unique talents and perspectives, contributing to the unraveling of the cosmic tapestry.

# They encountered celestial beings representing various aspects of existence—creators of galaxies, weavers of time, and guardians of cosmic balance. Through interactions with these beings, Seraphina and her companions began to understand the intricate threads that bound the realms together.

# The cosmic journey led them to the Celestial Archive, a repository of cosmic knowledge. Within its crystalline halls, they discovered ancient texts that chronicled the history of Eldoria and the enchanted dimensions, revealing the symbiotic relationship that had existed since time immemorial.

# It became clear that the celestial convergence was not an isolated event but a recurring cosmic phenomenon, designed to remind the realms of their shared destiny. Each convergence had woven new threads into the cosmic tapestry, enriching the interconnectedness of all existence.

# As they delved deeper into the Celestial Archive, Seraphina and her companions unearthed a revelation—a prophecy that foretold of a chosen one who would unlock the true potential of the celestial convergence and bridge the realms in a way never before achieved.

# Seraphina, with her innate connection to the enchanted realms, realized that she was the chosen one—the artisan whose creativity and vision could transcend the boundaries of time and dimensions. It was her destiny to weave the threads of the cosmic tapestry, fusing Eldoria and the enchanted dimensions into a harmonious unity.

# With newfound determination, Seraphina and her companions returned to the cosmic convergence gateway. As they stepped back into the city of Lumaria, they carried the knowledge of the cosmic tapestry's intricacies and the realization that the destiny of their interconnected worlds rested in their hands.

# Eldoria, bathed in the radiance of the celestial convergence, stood poised for a transformative moment—a moment that would redefine the realms' relationship and unveil the true potential of their shared destiny.

# Part 8: The Celestial Tapestry Woven

# Back in Lumaria, Seraphina and her companions returned with newfound knowledge and purpose. The city shimmered with anticipation, its crystal spires casting prismatic reflections of the cosmic convergence that had unfolded in the cosmic realm.

# As the chosen one, Seraphina stepped forward to share the revelations from the Celestial Archive. Eldorians and beings from the enchanted dimensions gathered in the heart of Lumaria, their hearts brimming with hope and curiosity.

# Seraphina's words resonated like a celestial melody, weaving a narrative that revealed the interconnectedness of Eldoria and the enchanted realms. She spoke of the cosmic convergence as a recurring phenomenon—a cosmic reminder that the destinies of their worlds were intertwined.

# The citizens of Lumaria and their counterparts from the enchanted dimensions listened with rapt attention, recognizing the profound implications of the cosmic revelations. It became clear that the celestial convergence held the key to a deeper unity—a unity that transcended the boundaries of time and dimensions.

# To unlock the true potential of the celestial convergence, Seraphina proposed a grand endeavor—a convergence ceremony that would fuse the realms in a way never before attempted. Eldorians and beings from the enchanted dimensions would come together to participate in this momentous event.

# Preparations for the convergence ceremony began, with artisans, scholars, and magicians collaborating to create a bridge that would unite the realms. The celestial convergence gateway, once ephemeral, was transformed into a permanent structure, pulsating with the cosmic energy.

# As the day of the convergence ceremony approached, Lumaria buzzed with activity. The crystal spires of the city resonated with the anticipation of a harmonious future, and the citizens felt a profound connection with their counterparts from the enchanted dimensions.

# The convergence ceremony, held beneath the twin moons of Eldoria, was a breathtaking spectacle. Eldorians and beings from the enchanted dimensions gathered at the cosmic nexus, their hearts aligned with the rhythm of the cosmic tapestry.

# Seraphina, as the chosen one, stood at the center of the convergence, her sculptures surrounding her. Each sculpture represented a facet of existence—a celestial constellation, an enchanted realm, and the boundless potential of unity.

# As the participants joined hands and focused their intentions, a radiant surge of energy coursed through the convergence gateway. The realms began to merge, their energies intertwining like threads in a cosmic tapestry.

# The skies above Lumaria shimmered with celestial colors, and the city itself seemed to transcend its physical form, existing simultaneously in multiple dimensions. Eldorians and beings from the enchanted dimensions felt a profound sense of unity as the convergence ceremony reached its zenith.

# In that transcendent moment, the celestial tapestry was woven anew, fusing Eldoria and the enchanted realms into a harmonious whole. The cosmic convergence, once a mystery, had become a transformative force, shaping the destiny of their interconnected worlds.

# As the convergence ceremony concluded, the citizens of Lumaria and their counterparts from the enchanted dimensions celebrated the unity they had forged. Seraphina's sculptures, now infused with the essence of the convergence, emanated a luminous glow, serving as a reminder of their shared destiny.

# Eldoria and the enchanted dimensions flourished as one, their destinies forever intertwined by the threads of the cosmic tapestry. The echoes of their cosmic journey would resound through the ages, inspiring future generations to embrace the power of unity, creativity, and the enduring magic that bound their worlds together.

# Part 9: Guardians of the Celestial Bond

# In the wake of the convergence ceremony, Eldoria and the enchanted dimensions stood united, their destinies forever interwoven by the cosmic tapestry. Lumaria, once a city within Eldoria, now existed as a nexus, simultaneously connected to multiple realms.

# Seraphina, the chosen one, emerged as a symbol of unity and creativity. Her sculptures continued to inspire, reflecting the harmonious blend of Eldoria and the enchanted dimensions. She worked tirelessly to nurture the bonds between their worlds, fostering an era of collaboration and exploration.

# The crystal spires of Lumaria, infused with the essence of the convergence, cast prismatic reflections that illuminated the city and its residents. Eldorians and beings from the enchanted dimensions coexisted in harmony, sharing knowledge, artistry, and the joys of discovery.

# The cosmic convergence had brought forth a golden age of enlightenment. Scholars and dreamers from all realms flocked to Lumaria to exchange ideas and explore the mysteries of existence. The city's libraries, now repositories of knowledge spanning dimensions, became a source of inspiration for generations to come.

# The celestial beings encountered in the cosmic realm remained as guardians of the celestial bond. They watched over Eldoria and the enchanted dimensions, guiding those who sought to deepen their understanding of the cosmic tapestry.

# With time, the cosmic convergence itself evolved, occurring at intervals that allowed the realms to continue their separate journeys while reaffirming their shared destiny. Each convergence added new threads to the celestial tapestry, enriching the unity of their worlds.

# The bond between Seraphina and Astral persisted, transcending the boundaries of time and dimensions. Astral, from the distant reaches of the cosmos, continued to offer guidance and wisdom, ensuring that the celestial harmony remained unbroken.

# Eldoria and the enchanted dimensions became a testament to the enduring power of unity and creativity. The cosmic journey that had brought them together became a timeless legend—a story that resonated in the hearts of all who encountered the radiant glow of their interconnected worlds.

# As the ages passed, the legacy of the cosmic convergence echoed through time and dimensions, inspiring future generations to embrace the transformative potential of harmony, creativity, and the enduring magic that bound their realms together.

# And so, in the heart of Lumaria, where starlight and crystal spires met, the tale of Seraphina, the chosen one, and the cosmic convergence became a timeless legend—a saga that celebrated the beauty of unity, the boundless possibilities of creativity, and the enduring power of the celestial tapestry.

# Part 10: A Cosmic Promise

# The ages continued to pass, and the cosmic bond between Eldoria and the enchanted dimensions endured. Lumaria, the nexus of unity, stood as a testament to the enduring power of their shared destiny.

# Seraphina, the chosen one, continued to sculpt celestial visions that celebrated the beauty of harmony and creativity. Her art not only adorned the crystal spires of Lumaria but also resonated through the realms, inspiring generations of dreamers, artists, and scholars.

# Eldoria's libraries, enriched by knowledge from the enchanted dimensions, became a hub of learning and collaboration. Scholars from various realms gathered to unravel the mysteries of existence, further deepening the bond between their worlds.

# The cosmic convergence, occurring at precise intervals, continued to be a cosmic reminder—a moment when Eldoria and the enchanted dimensions came together to weave new threads into the celestial tapestry. Each convergence revealed untold wonders and expanded the realms' understanding of their shared destiny.

# The guardians of the celestial bond, the celestial beings Seraphina and her companions had encountered in the cosmic realm, remained vigilant. They watched over the realms, guiding those who sought to explore the harmonious connection between their worlds.

# Astral, from his celestial vantage point, continued to offer wisdom and inspiration to Seraphina and the inhabitants of Eldoria. Their connection remained unbroken, a testament to the enduring friendship that transcended time and dimensions.

# Eldoria and the enchanted dimensions lived in unity and prosperity, their destinies forever intertwined by the threads of the cosmic tapestry. The echoes of their cosmic journey became a source of hope and inspiration for all who ventured to Lumaria and explored the realms beyond.

# As the ages passed, the legacy of the cosmic convergence lived on, inspiring future generations to embrace the transformative potential of unity, creativity, and the enduring magic that bound their realms together.

# And so, in the heart of Lumaria, where starlight and crystal spires met, the tale of Seraphina, the chosen one, and the cosmic convergence continued to be a timeless legend—a story that celebrated the beauty of unity, the boundless possibilities of creativity, and the enduring power of the celestial promise.

# If there are any specific themes, plot elements, or details you'd like to explore further or if you have any other requests for the story, please let me know, and I'll continue with the narrative.
# """]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    # True,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    # True,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False)
                                    )
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = OptLM(opt_config, env, args.path, policy)

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    # if DUMMY_WEIGHT not in args.path:
    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #     show_str = "Outputs:\n" + 70 * '-' + "\n"
    #     for i in [0, len(outputs)-1]:
    #         show_str += f"{i}: {outputs[i]}\n"
    #         show_str += "-" * 70 + "\n"
    #     if args.verbose >= 2:
    #         print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-66b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/mnt/nvme0n1-new-lv/data/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="/mnt/nvme0n1-new-lv/data/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=8)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[0, 100, 0, 100, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
