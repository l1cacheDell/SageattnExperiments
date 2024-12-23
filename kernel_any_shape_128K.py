import torch
from flash_attn.utils.benchmark import benchmark_forward
from flash_attn import flash_attn_func
from sageattention import sageattn
from rich import print
import sageattention._qattn as qattn
import torch.nn.functional as F
import argparse

WARP_Q = 32
WARP_K = 64

parser = argparse.ArgumentParser(description='Benchmark Baseline')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=4096, help="Sequence length")
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')

args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
target_len = args.seq_len
_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

kernel = qattn.qk_int8_sv_f16_accum_f16_attn

print(f"batch: {batch}, head: {head}, headdim: {headdim}")

def precision_diff(quant_o: torch.Tensor, fa2_o: torch.Tensor):
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    return sim, l1

is_causal = False
_is_causal = 1 if is_causal else 0
_repeat = 10
print(f"is_causal: {is_causal}")
for seq_len in {target_len}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    # flash attn
    q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    o = None
    for i in range(5): o = flash_attn_func(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    _, time1 = benchmark_forward(flash_attn_func, q, k, v, causal=is_causal, repeats=_repeat, verbose=False, desc='Triton')
    fa2_flops = flops/time1.mean*1e-12

    print("=========== Time Eval ===========")
    print(f"fa2 total: {time1.mean * _repeat}")
    print(f"fa2 time: {time1.mean}")
    
    # sage attn
    sm_scale = 1 / (headdim ** 0.5)
    q = q.to(torch.int8)
    k = k.to(torch.int8)
    v = v.to(torch.float16)

    q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float).cuda()
    k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float).cuda()
    o2 = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16).cuda()
    for i in range(5): kernel(q, k, v, o2, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
    
    torch.cuda.synchronize()
    _, time2 = benchmark_forward(kernel, q, k, v, o2, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=_repeat, verbose=False, desc='Triton')
    
    sa_flops = flops/time2.mean*1e-12
    print(f"sa  total: {time2.mean * _repeat}")
    print(f"sa  time: {time2.mean}")

    print("=========== ACCURACY ===========")
    sim, l1 = precision_diff(o2, o)
    print("cos sim: ", sim)
    print("L1 diff", l1)
    print("=========== FLOPS ===========")
    print(f"seq_len: {seq_len}, FA2 flops: {fa2_flops}")
    print(f'seq_len: {seq_len}, SA  flops: {sa_flops}')
    print(f"The speed up is: {sa_flops / fa2_flops} x")

print("\n\n======================\n\n")

is_causal = True
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {target_len}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    # flash attn
    q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    o = None
    for i in range(5): o = flash_attn_func(q, k, v, causal=is_causal)
    torch.cuda.synchronize()
    _, time1 = benchmark_forward(flash_attn_func, q, k, v, causal=is_causal, repeats=100, verbose=False, desc='Triton')
    fa2_flops = flops/time1.mean*1e-12
    print("=========== Time Eval ===========")
    print(f"fa2 total: {time1.mean * 100}")
    print(f"fa2 time: {time1.mean}")
    
    # sage attn
    # q = torch.randn(batch, seq_len, head, headdim).half().cuda()
    # k = torch.randn(batch, seq_len, head, headdim).half().cuda()
    # v = torch.randn(batch, seq_len, head, headdim).half().cuda()
    q = q.to(torch.int8)
    k = k.to(torch.int8)

    q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float).cuda()
    k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float).cuda()
    o2 = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16).cuda()
    for i in range(5): kernel(q, k, v, o2, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
    
    torch.cuda.synchronize()
    _, time2 = benchmark_forward(kernel, q, k, v, o2, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
    
    sa_flops = flops/time2.mean*1e-12
    print(f"sa total: {time2.mean * 100}")
    print(f"sa  time: {time2.mean}")
    
    print("=========== ACCURACY ===========")
    sim, l1 = precision_diff(o2, o)
    print("cos sim: ", sim)
    print("L1 diff", l1)
    print("=========== FLOPS ===========")
    print(f"seq_len: {seq_len}, FA2 flops: {fa2_flops}")
    print(f'seq_len: {seq_len}, SA  flops: {sa_flops}')
    print(f"The speed up is: {sa_flops / fa2_flops} x")