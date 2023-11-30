 __global__ __launch_bounds__(256, 4) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_vfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_vfuse_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(512, 0) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 384;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,384;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,384;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(512, 0) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 256;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(512, 0) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_idx_2(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,384;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,384;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 128;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(512, 4) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 384;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,384;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,384;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(512, 4) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_lb_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 256;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(512, 4) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_lb_idx_2(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    float B_local3[64];
    static float Apad_shared4[3072] __attribute__((shared));
    static float W_shared5[3072] __attribute__((shared));
    float Apad_shared_local6[8];
    float W_shared_local7[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local3[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local3[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 1,384;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(Apad_shared4 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A0 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_0) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_0) < 128) {
                        *(float4 *)(W_shared5 + ((((int)threadIdx_x_0) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W2 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_0) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 1,384;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3)];
                        Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + ((((int)threadIdx_x_0) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1)];
                        W_shared_local7[(ax3_1 + 4)] = W_shared5[((((rc_inner * 128) + ((((int)threadIdx_x_0) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 32)] = (B_local3[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local6[nn_c] * W_shared_local7[(ff_c + 4)]));
                            B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                            B_local3[(((ff_c * 4) + nn_c) + 48)] = (B_local3[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B1[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_0) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_0) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 128;
    float B_local11[64];
    static float Apad_shared12[3072] __attribute__((shared));
    static float W_shared13[3072] __attribute__((shared));
    float Apad_shared_local14[8];
    float W_shared_local15[8];
    for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
            B_local11[((ff_c_init * 4) + nn_c_init)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
            B_local11[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
        }
    }
    for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(Apad_shared12 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4 *)(A8 + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx_x_1) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                    }
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    if (((int)threadIdx_x_1) < 128) {
                        *(float4 *)(W_shared13 + ((((int)threadIdx_x_1) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx_x_1) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                    }
                }
                asm ("bar.sync 2,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 128) + ((((int)threadIdx_x_1) & 15) * 4)) + ax3) + 64)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 128) + ((((int)threadIdx_x_1) >> 4) * 4)) + ax3_1) + 64)];
                    }
                    for (int ff_c = 0; ff_c < 4; ++ff_c) {
                        for (int nn_c = 0; nn_c < 4; ++nn_c) {
                            B_local11[((ff_c * 4) + nn_c)] = (B_local11[((ff_c * 4) + nn_c)] + (Apad_shared_local14[nn_c] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 32)] = (B_local11[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local14[nn_c] * W_shared_local15[(ff_c + 4)]));
                            B_local11[(((ff_c * 4) + nn_c) + 16)] = (B_local11[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[ff_c]));
                            B_local11[(((ff_c * 4) + nn_c) + 48)] = (B_local11[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local14[(nn_c + 4)] * W_shared_local15[(ff_c + 4)]));
                        }
                    }
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B9[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx_x_1) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx_x_1) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
