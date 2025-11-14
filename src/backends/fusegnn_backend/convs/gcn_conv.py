import torch

from ..functional import coo2csr, csr2csc, fused_gar_agg, fused_gas_agg, gcn_gar_edge_weight, gcn_gas_edge_weight

# the final fused version


class garGCNConv(torch.nn.Module):
    def __init__(self, cached=False, flow="target_to_source"):
        r"""
        Args:
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
        """
        super(garGCNConv, self).__init__()
        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.tid, self.sid = (0, 1) if self.flow == "target_to_source" else (1, 0)

        self.cached = cached

        self.cached_tar_ptr = None
        self.cached_src_index = None
        self.cached_edge_weight_f = None
        self.cached_src_ptr = None
        self.cached_tar_index = None
        self.cached_edge_weight_b = None

        self.cached_num_edges = None

    def forward(
        self,
        x,
        edge_index=None,
        edge_weight=None,
        tar_ptr=None,
        src_index=None,
        src_ptr=None,
        tar_index=None,
        edge_weight_b=None,
    ):
        if not self.cached or self.cached_num_edges is None:
            # when the results are not cached, or it is the first execution.
            if tar_ptr is not None:  # when the CSR & CSC format are provided
                self.cached_tar_ptr = tar_ptr
                self.cached_src_index = src_index
                self.cached_edge_weight_f = edge_weight
                self.cached_src_ptr = src_ptr
                self.cached_tar_index = tar_index
                self.cached_edge_weight_b = edge_weight_b

                self.cached_num_edges = tar_ptr.size(0)
            else:
                num_nodes = x.size(0)
                # convert the edge lists to int32
                edge_index = edge_index.to(torch.int32)
                src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])

                # convert coo format to csr format
                self.cached_src_index, tar_index, self.cached_tar_ptr, edge_weight_f = coo2csr(
                    src_index, tar_index, num_nodes, edge_weight, False
                )

                # update edge weight
                self.cached_edge_weight_f = gcn_gar_edge_weight(
                    self.cached_src_index, self.cached_tar_ptr, tar_index, num_nodes, edge_weight_f, self.flow
                )

                # get the csc format for backward pass
                self.cached_src_ptr, self.cached_tar_index, self.cached_edge_weight_b = csr2csc(
                    self.cached_tar_ptr, self.cached_src_index, self.cached_edge_weight_f, num_nodes
                )

                self.cached_num_edges = self.cached_tar_ptr.size(0)

        return fused_gar_agg(
            feature=x,
            src_index=self.cached_src_index,
            tar_ptr=self.cached_tar_ptr,
            edge_weight_f=self.cached_edge_weight_f,
            tar_index=self.cached_tar_index,
            src_ptr=self.cached_src_ptr,
            edge_weight_b=self.cached_edge_weight_b,
            require_edge_weight=False,
        )


# the GAS version


class gasGCNConv(torch.nn.Module):
    def __init__(self, cached=False, flow="target_to_source"):
        r"""
        Args:
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
        """
        super(gasGCNConv, self).__init__()
        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.tid, self.sid = (0, 1) if self.flow == "target_to_source" else (1, 0)

        self.cached = cached
        self.cached_num_edges = None

        self.cached_src_index = None
        self.cached_tar_index = None
        self.cached_edge_weight = None

    def forward(self, x, edge_index=None, edge_weight=None, src_index=None, tar_index=None):
        if not self.cached or self.cached_num_edges is None:
            num_nodes = x.size(0)
            edge_index_int = edge_index.to(torch.int32)

            self.cached_src_index, self.cached_tar_index = (edge_index_int[self.sid], edge_index_int[self.tid])

            # Make them contiguous!
            self.cached_src_index = self.cached_src_index.contiguous()
            self.cached_tar_index = self.cached_tar_index.contiguous()

            self.cached_edge_weight = gcn_gas_edge_weight(
                self.cached_src_index, self.cached_tar_index, num_nodes, edge_weight, self.flow
            )

            self.cached_num_edges = self.cached_src_index.size(0)

        return fused_gas_agg(
            feature=x,
            src_index=self.cached_src_index,
            tar_index=self.cached_tar_index,
            edge_weight=self.cached_edge_weight,
            require_edge_weight=False,
        )
