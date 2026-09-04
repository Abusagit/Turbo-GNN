KERNELS="regex:GraphAttentionForward_CSR_MH_v2_D|compute_D_mh_kernel_D|graph_attn_backward_csrT_kernel_D|GATv2Forward_Kernel|GATv2Backward_AL|GATv2Backward_R|ReduceGradAKernel|reduction_aggr_backward_typed|reduction_aggr_forward_light_kernel_1d|reduction_aggr_forward_heavy_kernel|unpack_results_kernel|reduction_aggr_forward_heavy_kernel_2d"


# can launch without sudo

sudo CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-13.2/bin/ncu --kernel-name $KERNELS -f -o gt-fp32-fwd .venv/bin/python  scripts/benchmark.py --layer gt --backend cuda   --mode forward --warmup 0 --iters 1 --exact-iters
