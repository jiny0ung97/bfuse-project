 __global__ __launch_bounds__(128, 8) void bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_vfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float T_batch_matmul_NT_local3[32];
    static float A_shared4[4096] __attribute__((shared));
    static float B_shared5[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local3[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 1,128;");
        ;
        *(float2 *)(A_shared4 + (((int)threadIdx_x_0) * 2)) = *(float2 *)(A0 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 256)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 8192));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 512)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 16384));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 768)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 24576));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1024)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 32768));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1280)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 40960));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1536)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 49152));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1792)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 57344));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2048)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 65536));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2304)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 73728));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2560)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 81920));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2816)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 90112));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3072)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 98304));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3328)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 106496));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3584)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 114688));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3840)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 122880));
        B_shared5[((int)threadIdx_x_0)] = B1[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31))];
        B_shared5[(((int)threadIdx_x_0) + 128)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 4096)];
        B_shared5[(((int)threadIdx_x_0) + 256)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 8192)];
        B_shared5[(((int)threadIdx_x_0) + 384)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 12288)];
        B_shared5[(((int)threadIdx_x_0) + 512)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 16384)];
        B_shared5[(((int)threadIdx_x_0) + 640)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 20480)];
        B_shared5[(((int)threadIdx_x_0) + 768)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 24576)];
        B_shared5[(((int)threadIdx_x_0) + 896)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 28672)];
        asm ("bar.sync 1,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT2[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local3[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT2[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local3[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float T_batch_matmul_NT_local9[32];
    static float A_shared10[4096] __attribute__((shared));
    static float B_shared11[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local9[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 2,128;");
        ;
        *(float2 *)(A_shared10 + (((int)threadIdx_x_1) * 2)) = *(float2 *)(A6 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 256)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 8192));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 512)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 16384));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 768)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 24576));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1024)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 32768));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1280)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 40960));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1536)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 49152));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1792)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 57344));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2048)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 65536));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2304)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 73728));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2560)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 81920));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2816)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 90112));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3072)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 98304));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3328)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 106496));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3584)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 114688));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3840)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 122880));
        B_shared11[((int)threadIdx_x_1)] = B7[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31))];
        B_shared11[(((int)threadIdx_x_1) + 128)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 4096)];
        B_shared11[(((int)threadIdx_x_1) + 256)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 8192)];
        B_shared11[(((int)threadIdx_x_1) + 384)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 12288)];
        B_shared11[(((int)threadIdx_x_1) + 512)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 16384)];
        B_shared11[(((int)threadIdx_x_1) + 640)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 20480)];
        B_shared11[(((int)threadIdx_x_1) + 768)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 24576)];
        B_shared11[(((int)threadIdx_x_1) + 896)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 28672)];
        asm ("bar.sync 2,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT8[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local9[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT8[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local9[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
}
 __global__ __launch_bounds__(128, 0) void bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_vfuse_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float T_batch_matmul_NT_local3[32];
    static float A_shared4[4096] __attribute__((shared));
    static float B_shared5[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local3[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 1,128;");
        ;
        *(float2 *)(A_shared4 + (((int)threadIdx_x_0) * 2)) = *(float2 *)(A0 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 256)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 8192));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 512)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 16384));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 768)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 24576));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1024)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 32768));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1280)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 40960));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1536)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 49152));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1792)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 57344));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2048)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 65536));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2304)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 73728));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2560)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 81920));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2816)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 90112));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3072)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 98304));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3328)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 106496));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3584)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 114688));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3840)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 122880));
        B_shared5[((int)threadIdx_x_0)] = B1[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31))];
        B_shared5[(((int)threadIdx_x_0) + 128)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 4096)];
        B_shared5[(((int)threadIdx_x_0) + 256)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 8192)];
        B_shared5[(((int)threadIdx_x_0) + 384)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 12288)];
        B_shared5[(((int)threadIdx_x_0) + 512)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 16384)];
        B_shared5[(((int)threadIdx_x_0) + 640)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 20480)];
        B_shared5[(((int)threadIdx_x_0) + 768)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 24576)];
        B_shared5[(((int)threadIdx_x_0) + 896)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 28672)];
        asm ("bar.sync 1,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT2[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local3[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT2[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local3[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float T_batch_matmul_NT_local9[32];
    static float A_shared10[4096] __attribute__((shared));
    static float B_shared11[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local9[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 2,128;");
        ;
        *(float2 *)(A_shared10 + (((int)threadIdx_x_1) * 2)) = *(float2 *)(A6 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 256)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 8192));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 512)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 16384));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 768)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 24576));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1024)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 32768));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1280)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 40960));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1536)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 49152));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1792)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 57344));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2048)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 65536));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2304)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 73728));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2560)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 81920));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2816)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 90112));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3072)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 98304));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3328)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 106496));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3584)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 114688));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3840)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 122880));
        B_shared11[((int)threadIdx_x_1)] = B7[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31))];
        B_shared11[(((int)threadIdx_x_1) + 128)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 4096)];
        B_shared11[(((int)threadIdx_x_1) + 256)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 8192)];
        B_shared11[(((int)threadIdx_x_1) + 384)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 12288)];
        B_shared11[(((int)threadIdx_x_1) + 512)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 16384)];
        B_shared11[(((int)threadIdx_x_1) + 640)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 20480)];
        B_shared11[(((int)threadIdx_x_1) + 768)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 24576)];
        B_shared11[(((int)threadIdx_x_1) + 896)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 28672)];
        asm ("bar.sync 2,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT8[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local9[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT8[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local9[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_hfuse_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float T_batch_matmul_NT_local3[32];
    static float A_shared4[4096] __attribute__((shared));
    static float B_shared5[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local3[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 1,128;");
        ;
        *(float2 *)(A_shared4 + (((int)threadIdx_x_0) * 2)) = *(float2 *)(A0 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 256)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 8192));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 512)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 16384));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 768)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 24576));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1024)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 32768));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1280)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 40960));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1536)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 49152));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1792)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 57344));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2048)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 65536));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2304)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 73728));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2560)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 81920));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2816)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 90112));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3072)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 98304));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3328)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 106496));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3584)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 114688));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3840)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 122880));
        B_shared5[((int)threadIdx_x_0)] = B1[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31))];
        B_shared5[(((int)threadIdx_x_0) + 128)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 4096)];
        B_shared5[(((int)threadIdx_x_0) + 256)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 8192)];
        B_shared5[(((int)threadIdx_x_0) + 384)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 12288)];
        B_shared5[(((int)threadIdx_x_0) + 512)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 16384)];
        B_shared5[(((int)threadIdx_x_0) + 640)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 20480)];
        B_shared5[(((int)threadIdx_x_0) + 768)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 24576)];
        B_shared5[(((int)threadIdx_x_0) + 896)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 28672)];
        asm ("bar.sync 1,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT2[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local3[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT2[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local3[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    float T_batch_matmul_NT_local9[32];
    static float A_shared10[4096] __attribute__((shared));
    static float B_shared11[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local9[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 2,128;");
        ;
        *(float2 *)(A_shared10 + (((int)threadIdx_x_1) * 2)) = *(float2 *)(A6 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 256)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 8192));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 512)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 16384));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 768)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 24576));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1024)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 32768));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1280)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 40960));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1536)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 49152));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1792)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 57344));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2048)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 65536));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2304)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 73728));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2560)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 81920));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2816)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 90112));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3072)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 98304));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3328)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 106496));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3584)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 114688));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3840)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 122880));
        B_shared11[((int)threadIdx_x_1)] = B7[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31))];
        B_shared11[(((int)threadIdx_x_1) + 128)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 4096)];
        B_shared11[(((int)threadIdx_x_1) + 256)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 8192)];
        B_shared11[(((int)threadIdx_x_1) + 384)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 12288)];
        B_shared11[(((int)threadIdx_x_1) + 512)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 16384)];
        B_shared11[(((int)threadIdx_x_1) + 640)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 20480)];
        B_shared11[(((int)threadIdx_x_1) + 768)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 24576)];
        B_shared11[(((int)threadIdx_x_1) + 896)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 28672)];
        asm ("bar.sync 2,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT8[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local9[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT8[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local9[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
}
 __global__ __launch_bounds__(256, 8) void bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float T_batch_matmul_NT_local3[32];
    static float A_shared4[4096] __attribute__((shared));
    static float B_shared5[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local3[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local3[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 1,128;");
        ;
        *(float2 *)(A_shared4 + (((int)threadIdx_x_0) * 2)) = *(float2 *)(A0 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 256)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 8192));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 512)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 16384));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 768)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 24576));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1024)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 32768));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1280)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 40960));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1536)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 49152));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 1792)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 57344));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2048)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 65536));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2304)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 73728));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2560)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 81920));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 2816)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 90112));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3072)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 98304));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3328)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 106496));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3584)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 114688));
        *(float2 *)(A_shared4 + ((((int)threadIdx_x_0) * 2) + 3840)) = *(float2 *)(A0 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + 122880));
        B_shared5[((int)threadIdx_x_0)] = B1[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31))];
        B_shared5[(((int)threadIdx_x_0) + 128)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 4096)];
        B_shared5[(((int)threadIdx_x_0) + 256)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 8192)];
        B_shared5[(((int)threadIdx_x_0) + 384)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 12288)];
        B_shared5[(((int)threadIdx_x_0) + 512)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 16384)];
        B_shared5[(((int)threadIdx_x_0) + 640)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 20480)];
        B_shared5[(((int)threadIdx_x_0) + 768)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 24576)];
        B_shared5[(((int)threadIdx_x_0) + 896)] = B1[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_0) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_0) & 31)) + 28672)];
        asm ("bar.sync 1,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local3[(i_c_outer_inner * 8)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 16)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 1)] + (A_shared4[((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 17)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 2)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 18)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 3)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 19)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 4)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 20)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 5)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 21)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 6)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 22)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[(((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 7)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local3[((i_c_outer_inner * 8) + 23)] + (A_shared4[(((((((int)threadIdx_x_0) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared5[((((((int)threadIdx_x_0) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT2[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local3[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT2[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_0) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_0) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local3[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    float T_batch_matmul_NT_local9[32];
    static float A_shared10[4096] __attribute__((shared));
    static float B_shared11[1024] __attribute__((shared));
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
        T_batch_matmul_NT_local9[(i_c_outer_inner_init * 8)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 16)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 1)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 17)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 2)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 18)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 3)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 19)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 4)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 20)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 5)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 21)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 6)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 22)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 7)] = 0.F;
        T_batch_matmul_NT_local9[((i_c_outer_inner_init * 8) + 23)] = 0.F;
    }
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 2,128;");
        ;
        *(float2 *)(A_shared10 + (((int)threadIdx_x_1) * 2)) = *(float2 *)(A6 + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 256)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 8192));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 512)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 16384));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 768)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 24576));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1024)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 32768));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1280)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 40960));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1536)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 49152));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 1792)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 57344));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2048)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 65536));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2304)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 73728));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2560)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 81920));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 2816)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 90112));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3072)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 98304));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3328)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 106496));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3584)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 114688));
        *(float2 *)(A_shared10 + ((((int)threadIdx_x_1) * 2) + 3840)) = *(float2 *)(A6 + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + 122880));
        B_shared11[((int)threadIdx_x_1)] = B7[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31))];
        B_shared11[(((int)threadIdx_x_1) + 128)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 4096)];
        B_shared11[(((int)threadIdx_x_1) + 256)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 8192)];
        B_shared11[(((int)threadIdx_x_1) + 384)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 12288)];
        B_shared11[(((int)threadIdx_x_1) + 512)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 16384)];
        B_shared11[(((int)threadIdx_x_1) + 640)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 20480)];
        B_shared11[(((int)threadIdx_x_1) + 768)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 24576)];
        B_shared11[(((int)threadIdx_x_1) + 896)] = B7[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx_x_1) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx_x_1) & 31)) + 28672)];
        asm ("bar.sync 2,128;");
        ;
        for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
            for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local9[(i_c_outer_inner * 8)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 16)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 1)] + (A_shared10[((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 17)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 2)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 18)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 3)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 19)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 4)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 20)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 5)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 21)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 6)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 22)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[(((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 7)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
                T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local9[((i_c_outer_inner * 8) + 23)] + (A_shared10[(((((((int)threadIdx_x_1) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared11[((((((int)threadIdx_x_1) & 15) * 64) + k_outer_inner) + 32)]));
            }
        }
    }
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
            T_batch_matmul_NT8[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local9[((i_inner * 2) + j_inner)];
            T_batch_matmul_NT8[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx_x_1) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx_x_1) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local9[(((i_inner * 2) + j_inner) + 16)];
        }
    }
}
}
