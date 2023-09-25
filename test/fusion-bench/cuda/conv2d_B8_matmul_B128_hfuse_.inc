extern "C" __global__ __launch_bounds__(160, 0) void conv2d_B8_matmul_B128_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8)
{
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128) && (blockIdx.x < 392)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    float conv2d_nhwc_local3[16];
    static float pad_temp_shared4[480] __attribute__((shared));
    static float kernel_shared5[3072] __attribute__((shared));
    for (int nn_c_outer_inner_init = 0; nn_c_outer_inner_init < 2; ++nn_c_outer_inner_init) {
        for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
            for (int nn_c_inner_init = 0; nn_c_inner_init < 2; ++nn_c_inner_init) {
                conv2d_nhwc_local3[(((nn_c_outer_inner_init * 4) + (nn_c_inner_init * 2)) + ff_c_outer_inner_init)] = 0.F;
                conv2d_nhwc_local3[((((nn_c_outer_inner_init * 4) + (nn_c_inner_init * 2)) + ff_c_outer_inner_init) + 8)] = 0.F;
            }
        }
    }
    for (int ry_outer_outer = 0; ry_outer_outer < 3; ++ry_outer_outer) {
        for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
            asm ("bar.sync 1,128;");
            ;
            if (((int)threadIdx_x_0) < 120) {
                *(float4 *)(pad_temp_shared4 + (((int)threadIdx_x_0) * 4)) = (((1 <= (((((((int)blockIdx.x) % 196) / 14) * 4) + ((((int)threadIdx_x_0) % 30) / 10)) + ry_outer_outer)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx_x_0) % 10) >> 1)))) ? *(float4 *)(data0 + (((((((((((((int)blockIdx.x) / 196) * 802816) + ((((int)threadIdx_x_0) / 30) * 200704)) + (((((int)blockIdx.x) % 196) / 14) * 14336)) + (((((int)threadIdx_x_0) % 30) / 10) * 3584)) + (ry_outer_outer * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx_x_0) % 10) >> 1) * 64)) + (rc_outer_outer * 8)) + ((((int)threadIdx_x_0) & 1) * 4)) - 3648)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
                *(float4 *)(kernel_shared5 + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(kernel1 + (((((ry_outer_outer * 24576) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 8192)) + (rc_outer_outer * 1024)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 512)) + (((int)threadIdx_x_0) * 4)));
            }
            asm ("bar.sync 1,128;");
            ;
            for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
                for (int nn_c_outer_inner = 0; nn_c_outer_inner < 2; ++nn_c_outer_inner) {
                    for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
                        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
                            for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
                                for (int nn_c_inner = 0; nn_c_inner < 2; ++nn_c_inner) {
                                    conv2d_nhwc_local3[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] = (conv2d_nhwc_local3[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] + (pad_temp_shared4[((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx_x_0) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)] * kernel_shared5[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx_x_0) & 63) * 2)) + ff_c_outer_inner)]));
                                    conv2d_nhwc_local3[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] = (conv2d_nhwc_local3[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] + (pad_temp_shared4[(((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx_x_0) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner) + 16)] * kernel_shared5[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx_x_0) & 63) * 2)) + ff_c_outer_inner)]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int nn_inner = 0; nn_inner < 4; ++nn_inner) {
        for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
            conv2d_nhwc2[((((((((((int)blockIdx.x) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx.x) % 196) / 14) * 7168)) + ((((int)threadIdx_x_0) >> 6) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((int)threadIdx_x_0) & 63) * 2)) + ff_inner)] = conv2d_nhwc_local3[((nn_inner * 2) + ff_inner)];
            conv2d_nhwc2[(((((((((((int)blockIdx.x) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx.x) % 196) / 14) * 7168)) + ((((int)threadIdx_x_0) >> 6) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((int)threadIdx_x_0) & 63) * 2)) + ff_inner) + 128)] = conv2d_nhwc_local3[(((nn_inner * 2) + ff_inner) + 8)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160) && (blockIdx.x < 500)){
    unsigned int blockDim_x_1 = 32;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 32;
    float T_matmul_NT_local9[8];
    static float data_shared10[256] __attribute__((shared));
    static float weight_shared11[64] __attribute__((shared));
    T_matmul_NT_local9[0] = 0.F;
    T_matmul_NT_local9[2] = 0.F;
    T_matmul_NT_local9[4] = 0.F;
    T_matmul_NT_local9[6] = 0.F;
    T_matmul_NT_local9[1] = 0.F;
    T_matmul_NT_local9[3] = 0.F;
    T_matmul_NT_local9[5] = 0.F;
    T_matmul_NT_local9[7] = 0.F;
    for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
        asm ("bar.sync 2,32;");
        ;
        *(float4 *)(data_shared10 + (((int)threadIdx_x_1) * 4)) = *(float4 *)(data6 + (((((((int)blockIdx.x) / 125) * 16384) + ((((int)threadIdx_x_1) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx_x_1) & 1) * 4)));
        *(float4 *)(data_shared10 + ((((int)threadIdx_x_1) * 4) + 128)) = *(float4 *)(data6 + ((((((((int)blockIdx.x) / 125) * 16384) + ((((int)threadIdx_x_1) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx_x_1) & 1) * 4)) + 8192));
        *(float2 *)(weight_shared11 + (((int)threadIdx_x_1) * 2)) = *(float2 *)(weight7 + (((((((int)blockIdx.x) % 125) * 4096) + ((((int)threadIdx_x_1) >> 2) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx_x_1) & 3) * 2)));
        asm ("bar.sync 2,32;");
        ;
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[((((int)threadIdx_x_1) >> 1) * 8)] * weight_shared11[((((int)threadIdx_x_1) & 1) * 16)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[((((int)threadIdx_x_1) >> 1) * 8)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 32)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 128)] * weight_shared11[((((int)threadIdx_x_1) & 1) * 16)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 128)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 32)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[((((int)threadIdx_x_1) >> 1) * 8)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 8)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[((((int)threadIdx_x_1) >> 1) * 8)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 40)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 128)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 8)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 128)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 40)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 1)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 1)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 1)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 33)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 129)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 1)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 129)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 33)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 1)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 9)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 1)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 41)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 129)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 9)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 129)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 41)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 2)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 2)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 2)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 34)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 130)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 2)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 130)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 34)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 2)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 10)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 2)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 42)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 130)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 10)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 130)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 42)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 3)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 3)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 3)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 35)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 131)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 3)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 131)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 35)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 3)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 11)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 3)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 43)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 131)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 11)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 131)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 43)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 4)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 4)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 4)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 36)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 132)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 4)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 132)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 36)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 4)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 12)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 4)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 44)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 132)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 12)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 132)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 44)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 5)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 5)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 5)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 37)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 133)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 5)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 133)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 37)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 5)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 13)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 5)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 45)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 133)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 13)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 133)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 45)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 6)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 6)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 6)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 38)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 134)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 6)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 134)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 38)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 6)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 14)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 6)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 46)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 134)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 14)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 134)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 46)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 7)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 7)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 7)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 39)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 135)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 7)]));
        T_matmul_NT_local9[6] = (T_matmul_NT_local9[6] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 135)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 39)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 7)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 15)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 7)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 47)]));
        T_matmul_NT_local9[5] = (T_matmul_NT_local9[5] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 135)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 15)]));
        T_matmul_NT_local9[7] = (T_matmul_NT_local9[7] + (data_shared10[(((((int)threadIdx_x_1) >> 1) * 8) + 135)] * weight_shared11[(((((int)threadIdx_x_1) & 1) * 16) + 47)]));
    }
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
        T_matmul_NT8[((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx_x_1) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx_x_1) & 1) * 2)) + j_inner)] = T_matmul_NT_local9[j_inner];
        T_matmul_NT8[(((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx_x_1) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx_x_1) & 1) * 2)) + j_inner) + 4)] = T_matmul_NT_local9[(j_inner + 2)];
        T_matmul_NT8[(((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx_x_1) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx_x_1) & 1) * 2)) + j_inner) + 16000)] = T_matmul_NT_local9[(j_inner + 4)];
        T_matmul_NT8[(((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx_x_1) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx_x_1) & 1) * 2)) + j_inner) + 16004)] = T_matmul_NT_local9[(j_inner + 6)];
    }
}
}
