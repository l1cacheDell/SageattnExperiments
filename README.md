# Sage Attention Experiment field

As the `thu-ml` team released their sage attention kernels, many wants to test the performance of the sageattn kernel and make comparison with Flash Attention v2 or v3, while test scripts presented in the official repository `bench` is far not enough for us to get convincing results, thus I created a directory, to present some test scripts I used to compare with other kernels.

## `kernel_any_shape.py`: test any shape of your kernel

The official benchmark scripts just test the kernels in several specific kernels, like {1024, 4096, etc...}. When I was trying to put the qkv in another shape like (2, 24, 64, 4250), it throws the error:

```
    for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, sm_scale, 0)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Tensor query_scale must have shape (batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q)))
```

The reason is that they adopted `qk_int8_sv_f16_accum_f16_attn_per_warp` or `qk_int8_sv_f16_accum_f32_attn_per_warp_buf` or `qk_int8_sv_f16_accum_f32_attn_per_warp` three specific kernels which allowed the limited input shapes to specific sequence length.

If you want to test any shape you like, you should profile from the entrypoint: `sageattn` function.

So that's what `kernel_any_shape.py` does, it compares sage attn in any shape with FA2, use it:

```bash
python kernel_any_shape.py --batch_size 2 --num_heads 24 --head_dim 64 --seq_len 4250
```

The result will be:

```
batch: 4, head: 24, headdim: 64
is_causal: False
seq_len: 4250, FA2 flops: 181.51139511744935
seq_len: 4250, SA flops: 782.0451647811955
The speed up is: 4.308518284899759 x
is_causal: True
seq_len: 4250, FA2 flops: 152.92973555500123
seq_len: 4250, SA flops: 383.03959972225994
The speed up is: 2.5046770553297635 x
```

## `e2e_llama.py`: in process
The end to end performance test scripts hasn't been released yet and I just do some rough test on Llama-2-7B

This script needs improvement.

```bash
python e2e_llama.py
```