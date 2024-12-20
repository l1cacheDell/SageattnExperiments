import torch
from flash_attn.utils.benchmark import benchmark_forward
from flash_attn import flash_attn_func
from sageattention import sageattn

import argparse

parser = argparse.ArgumentParser(description='Benchmark Baseline')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=4096, help="Sequence length")
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
target_len = args.seq_len

print(f"batch: {batch}, head: {head}, headdim: {headdim}")

is_causal = False
print(f"is_causal: {is_causal}")
for seq_len in {target_len}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    # flash attn
    q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    for i in range(5): flash_attn_func(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    _, time1 = benchmark_forward(flash_attn_func, q, k, v, causal=is_causal, repeats=100, verbose=False, desc='Triton')
    fa2_flops = flops/time1.mean*1e-12
    
    # sage attn
    q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    sm_scale = 1 / (headdim ** 0.5)
    for i in range(5): sageattn(q, k, v, is_causal=is_causal, tensor_layout='NHD', sm_scale=sm_scale)
    torch.cuda.synchronize()
    _, time2 = benchmark_forward(sageattn, q, k, v, is_causal=is_causal, sm_scale=sm_scale, 
                                repeats=100, verbose=False, desc='Triton')
    sa_flops = flops/time2.mean*1e-12
    
    print(f"seq_len: {seq_len}, FA2 flops: {fa2_flops}")
    print(f'seq_len: {seq_len}, SA  flops: {sa_flops}')
    print(f"The speed up is: {sa_flops / fa2_flops} x")

is_causal = True
print(f"is_causal: {is_causal}")
for seq_len in {target_len}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    # flash attn
    q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    for i in range(5): flash_attn_func(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    _, time1 = benchmark_forward(flash_attn_func, q, k, v, causal=is_causal, repeats=100, verbose=False, desc='Triton')
    fa2_flops = flops/time1.mean*1e-12
    
    # sage attn
    q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    sm_scale = 1 / (headdim ** 0.5)
    for i in range(5): sageattn(q, k, v, is_causal=is_causal, tensor_layout='NHD', sm_scale=sm_scale)
    torch.cuda.synchronize()
    _, time2 = benchmark_forward(sageattn, q, k, v, is_causal=is_causal, sm_scale=sm_scale, 
                                repeats=100, verbose=False, desc='Triton')
    sa_flops = flops/time2.mean*1e-12
    
    print(f"seq_len: {seq_len}, FA2 flops: {fa2_flops}")
    print(f'seq_len: {seq_len}, SA  flops: {sa_flops}')
    print(f"The speed up is: {sa_flops / fa2_flops} x")