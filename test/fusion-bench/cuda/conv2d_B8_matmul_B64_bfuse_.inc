extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B64_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8)
{
if ((threadIdx.x < 128) && 
    (((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 0 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 84)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 168 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 252)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 336 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 420)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 504 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 588)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 672 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 728)))
{
    unsigned int others;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 0)
        others = 0;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 168)
        others = 84;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 336)
        others = 168;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 504)
        others = 252;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 672)
        others = 336;
    unsigned int gridDim_x_0 = 392;
    unsigned int blockIdx_x_0 = ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) - others) % 392;
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
            if (((int)threadIdx.x) < 120) {
                *(float4 *)(pad_temp_shared4 + (((int)threadIdx.x) * 4)) = (((1 <= (((((((int)blockIdx_x_0) % 196) / 14) * 4) + ((((int)threadIdx.x) % 30) / 10)) + ry_outer_outer)) && (1 <= (((((int)blockIdx_x_0) % 14) * 4) + ((((int)threadIdx.x) % 10) >> 1)))) ? *(float4 *)(data0 + (((((((((((((int)blockIdx_x_0) / 196) * 802816) + ((((int)threadIdx.x) / 30) * 200704)) + (((((int)blockIdx_x_0) % 196) / 14) * 14336)) + (((((int)threadIdx.x) % 30) / 10) * 3584)) + (ry_outer_outer * 3584)) + ((((int)blockIdx_x_0) % 14) * 256)) + (((((int)threadIdx.x) % 10) >> 1) * 64)) + (rc_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) - 3648)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
                *(float4 *)(kernel_shared5 + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)threadIdx.x) * 4))) = *(float4 *)(kernel1 + (((((ry_outer_outer * 24576) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 8192)) + (rc_outer_outer * 1024)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 512)) + (((int)threadIdx.x) * 4)));
            }
            asm ("bar.sync 1,128;");
            ;
            for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
                for (int nn_c_outer_inner = 0; nn_c_outer_inner < 2; ++nn_c_outer_inner) {
                    for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
                        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
                            for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
                                for (int nn_c_inner = 0; nn_c_inner < 2; ++nn_c_inner) {
                                    conv2d_nhwc_local3[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] = (conv2d_nhwc_local3[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] + (pad_temp_shared4[((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx.x) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)] * kernel_shared5[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx.x) & 63) * 2)) + ff_c_outer_inner)]));
                                    conv2d_nhwc_local3[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] = (conv2d_nhwc_local3[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] + (pad_temp_shared4[(((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx.x) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner) + 16)] * kernel_shared5[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx.x) & 63) * 2)) + ff_c_outer_inner)]));
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
            conv2d_nhwc2[((((((((((int)blockIdx_x_0) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx_x_0) % 196) / 14) * 7168)) + ((((int)threadIdx.x) >> 6) * 3584)) + ((((int)blockIdx_x_0) % 14) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + ff_inner)] = conv2d_nhwc_local3[((nn_inner * 2) + ff_inner)];
            conv2d_nhwc2[(((((((((((int)blockIdx_x_0) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx_x_0) % 196) / 14) * 7168)) + ((((int)threadIdx.x) >> 6) * 3584)) + ((((int)blockIdx_x_0) % 14) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + ff_inner) + 128)] = conv2d_nhwc_local3[(((nn_inner * 2) + ff_inner) + 8)];
        }
    }
}
if ((threadIdx.x < 32) && 
    (((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 84 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 168)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 252 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 336)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 420 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 504)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 588 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 672)
    || ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 728 && (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) < 792)))
{
    unsigned int others;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 84)
        others = 84;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 252)
        others = 168;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 420)
        others = 252;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 588)
        others = 336;
    if ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) >= 728)
        others = 392;
    unsigned int gridDim_x_1 = 400;
    unsigned int blockIdx_x_1 = ((blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) - others) % 400;
    float T_matmul_NT_local9[5];
    static float data_shared10[256] __attribute__((shared));
    static float weight_shared11[160] __attribute__((shared));
    T_matmul_NT_local9[0] = 0.F;
    T_matmul_NT_local9[1] = 0.F;
    T_matmul_NT_local9[2] = 0.F;
    T_matmul_NT_local9[3] = 0.F;
    T_matmul_NT_local9[4] = 0.F;
    for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
        asm ("bar.sync 2,32;");
        ;
        *(float2 *)(data_shared10 + (((int)threadIdx.x) * 2)) = *(float2 *)(data6 + (((((((int)blockIdx_x_1) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
        *(float2 *)(data_shared10 + ((((int)threadIdx.x) * 2) + 64)) = *(float2 *)(data6 + ((((((((int)blockIdx_x_1) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2048));
        *(float2 *)(data_shared10 + ((((int)threadIdx.x) * 2) + 128)) = *(float2 *)(data6 + ((((((((int)blockIdx_x_1) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096));
        *(float2 *)(data_shared10 + ((((int)threadIdx.x) * 2) + 192)) = *(float2 *)(data6 + ((((((((int)blockIdx_x_1) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 6144));
        weight_shared11[((int)threadIdx.x)] = weight7[(((((((int)blockIdx_x_1) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
        weight_shared11[(((int)threadIdx.x) + 32)] = weight7[((((((((int)blockIdx_x_1) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 1024)];
        weight_shared11[(((int)threadIdx.x) + 64)] = weight7[((((((((int)blockIdx_x_1) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 2048)];
        weight_shared11[(((int)threadIdx.x) + 96)] = weight7[((((((((int)blockIdx_x_1) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3072)];
        weight_shared11[(((int)threadIdx.x) + 128)] = weight7[((((((((int)blockIdx_x_1) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 4096)];
        asm ("bar.sync 2,32;");
        ;
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[((((int)threadIdx.x) >> 1) * 16)] * weight_shared11[((((int)threadIdx.x) & 1) * 16)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[((((int)threadIdx.x) >> 1) * 16)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 32)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[((((int)threadIdx.x) >> 1) * 16)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 64)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[((((int)threadIdx.x) >> 1) * 16)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 96)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[((((int)threadIdx.x) >> 1) * 16)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 128)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 1)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 33)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 65)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 97)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 129)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 2)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 34)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 66)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 98)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 130)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 3)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 35)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 67)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 99)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 131)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 4)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 36)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 68)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 100)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 132)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 5)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 37)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 69)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 101)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 133)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 6)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 38)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 70)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 102)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 134)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 7)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 39)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 71)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 103)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 135)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 8)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 40)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 72)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 104)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 136)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 9)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 41)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 73)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 105)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 137)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 10)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 42)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 74)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 106)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 138)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 11)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 43)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 75)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 107)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 139)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 12)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 44)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 76)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 108)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 140)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 13)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 45)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 77)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 109)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 141)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 14)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 46)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 78)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 110)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 142)]));
        T_matmul_NT_local9[0] = (T_matmul_NT_local9[0] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 15)]));
        T_matmul_NT_local9[1] = (T_matmul_NT_local9[1] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 47)]));
        T_matmul_NT_local9[2] = (T_matmul_NT_local9[2] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 79)]));
        T_matmul_NT_local9[3] = (T_matmul_NT_local9[3] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 111)]));
        T_matmul_NT_local9[4] = (T_matmul_NT_local9[4] + (data_shared10[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared11[(((((int)threadIdx.x) & 1) * 16) + 143)]));
    }
    T_matmul_NT8[(((((((int)blockIdx_x_1) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_1) % 100) * 10)) + (((int)threadIdx.x) & 1))] = T_matmul_NT_local9[0];
    T_matmul_NT8[((((((((int)blockIdx_x_1) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_1) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 2)] = T_matmul_NT_local9[1];
    T_matmul_NT8[((((((((int)blockIdx_x_1) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_1) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 4)] = T_matmul_NT_local9[2];
    T_matmul_NT8[((((((((int)blockIdx_x_1) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_1) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 6)] = T_matmul_NT_local9[3];
    T_matmul_NT8[((((((((int)blockIdx_x_1) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_1) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 8)] = T_matmul_NT_local9[4];
}
}
