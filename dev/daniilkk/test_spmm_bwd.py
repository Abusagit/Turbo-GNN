# %%
import unittest

import torch

from src.backends.cusparse_backend.utils import csr_SPMM_normalized

# %%
indptr = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="cuda")
indices = torch.tensor([1, 2, 0, 1, 2], dtype=torch.int32, device="cuda")
features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device="cuda")
# edge_weights = torch.tensor([0.5, 1.0, 0.5, 1.0, 0.5], dtype=torch.float32, device='cuda')

result = csr_SPMM_normalized(indptr, indices, features, norm="none", algorithm=3, use_cache=False, do_transpose_a=False)
# expected = torch.tensor([[1., 2.], [1., 0.], [1., 2.]], dtype=torch.float32, device='cuda')
# torch.testing.assert_close(result, expected)

# %%
csr_SPMM_normalized(indptr, indices, features, norm="none", algorithm=3, use_cache=False, do_transpose_a=True)
