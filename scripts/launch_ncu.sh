KERNELS="regex:GraphAttentionForward_CSR_MH_v2_D|compute_D_mh_kernel_D|graph_attn_backward_csrT_kernel_D|GATv2Forward_Kernel|GATv2Backward_AL|GATv2Backward_R|ReduceGradAKernel|reduction_aggr_backward_typed|reduction_aggr_forward_light_kernel_1d|reduction_aggr_forward_heavy_kernel|unpack_results_kernel|reduction_aggr_forward_heavy_kernel_2d"




CUDA_VISIBLE_DEVICES=4 /usr/local/cuda-12.6/bin/ncu --kernel-name $KERNELS -f -o gt-fp32  python scripts/benchmark.py --layer gt --backend cuda --dataset configs/datasets/main/ogbn_arxiv.yaml  --mode backward
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda-12.6/bin/ncu --kernel-name $KERNELS -f -o gt-bf16  python scripts/benchmark.py --layer gt --backend cuda --dataset configs/datasets/main/ogbn_arxiv.yaml  --mode backward --amp bf16


CUDA_VISIBLE_DEVICES=4 /usr/local/cuda-12.6/bin/ncu --kernel-name $KERNELS -f -o gatv2-fp32  python scripts/benchmark.py --layer gat_v2 --backend cuda --dataset configs/datasets/main/ogbn_arxiv.yaml  --mode backward
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda-12.6/bin/ncu --kernel-name $KERNELS -f -o gatv2-bf16  python scripts/benchmark.py --layer gat_v2 --backend cuda --dataset configs/datasets/main/ogbn_arxiv.yaml  --mode backward --amp bf16

CUDA_VISIBLE_DEVICES=4 /usr/local/cuda-12.6/bin/ncu --kernel-name $KERNELS -f -o minaggr-fp32  python scripts/benchmark.py --layer min_aggr --backend cuda --dataset configs/datasets/main/ogbn_arxiv.yaml  --mode backward
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda-12.6/bin/ncu --kernel-name $KERNELS -f -o minaggr-bf16  python scripts/benchmark.py --layer min_aggr --backend cuda --dataset configs/datasets/main/ogbn_arxiv.yaml  --mode backward --amp bf16
