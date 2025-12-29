#ifndef CONFIG_H
#define CONFIG_H

#define NUM_SM_GPU 56 // A30

#define BLK_H 16
#define BLK_W 8

#define BLK_M 16
#define BLK_N 8
#define BLK_K 16

#define WARP_SIZE 32
// for the new sddmm kernel where each warp computes this amount of TCBlocks of S
#define MIN_TCBLOCK_PER_WARP 1
// for the new sddmm kernel, now the number of warps does not depend on the embedding dimension
#define WARP_PER_TB 20

#define TCBLOCK_PER_WARP_FMM 2
// #define TCBLOCK_PER_WARP 32
#endif
