import cutlass
import cutlass.cute as cute


@cute.kernel
def transpose_kernel(self, mA: cute.Tensor, mB: cute.Tensor):
    tidx = cute.arch.thread_idx()[0]
    tidy = cute.arch.thread_idx()[1]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    # This might all be unnecessary
    # but I was fearful of the compiler
    tile_start_m = cutlass.Int32(0)
    tile_start_n = cutlass.Int32(0)
    global_m = cutlass.Int32(0)
    global_n = cutlass.Int32(0)
    M = cutlass.Int32(0)
    N = cutlass.Int32(0)
    val = cutlass.Float32(0.0)
    # Calculate tile starting positions
    tile_start_m = bidy * self._tile_size
    tile_start_n = bidx * self._tile_size
    # Calculate global coordinates for this thread
    global_m = tile_start_m + tidy
    global_n = tile_start_n + tidx
    # Get matrix dimensions at runtime
    M = mA.shape[0]
    N = mA.shape[1]
    # Bounds checking and transpose operation
    if global_m < M and global_n < N:
        val = mA[global_m, global_n]
        # Transpose: B[n, m] = A[m, n]
        mB[global_n, global_m] = val


@cute.jit  # host side
def launch(self, A: cute.Tensor, B: cute.Tensor):
    M, N = A.shape
    grid = ((N + self.T - 1) // self.T, (M + self.T - 1) // self.T, 1)
    self.transpose_kernel(A, B).launch(
        grid=grid,
        block=[self.T, self.T, 1],
    )


A_cute = from_dlpack(A).mark_layout_dynamic()
