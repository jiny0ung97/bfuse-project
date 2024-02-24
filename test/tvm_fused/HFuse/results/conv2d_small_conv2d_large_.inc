 __global__ __launch_bounds__(256, 5) void conv2d_small_conv2d_large_fused_kernel_vfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[1024] __attribute__((shared));
    static float W_shared5[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        asm ("bar.sync 1,256;");
        ;
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
            if (((int)threadIdx_y_0) < 8) {
                *(float4 *)(Apad_shared4 + (((((int)threadIdx_y_0) * 128) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4 *)(A0 + ((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_0) * 128)) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
        }
        if (((int)threadIdx_y_0) < 8) {
            *(float4 *)(W_shared5 + ((((int)threadIdx_y_0) * 64) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(W2 + (((rc_outer * 512) + (((int)threadIdx_y_0) * 64)) + (((int)threadIdx_x_0) * 4)));
        }
        asm ("bar.sync 1,256;");
        ;
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
            for (int ax3 = 0; ax3 < 4; ++ax3) {
                Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3)];
                Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3) + 64)];
            }
            for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 64) + (((int)threadIdx_y_0) * 4)) + ax3_1)];
            }
            for (int ff_c = 0; ff_c < 4; ++ff_c) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                    B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                    B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 16;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 16;
    unsigned int blockDim_y_1 = 8;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 16 % 8;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float B_local11[64];
    static float Apad_shared12[512] __attribute__((shared));
    static float W_shared13[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    *(float4 *)(Apad_shared12 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? *(float4 *)(A8 + ((((((((ry * 229376) + (((int)blockIdx.z) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    *(float4 *)(W_shared13 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4)));
                }
                asm ("bar.sync 2,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3) + 32)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1) + 32)];
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
            B9[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2048)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 32)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2080)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void conv2d_small_conv2d_large_fused_kernel_vfuse_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[1024] __attribute__((shared));
    static float W_shared5[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        asm ("bar.sync 1,256;");
        ;
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
            if (((int)threadIdx_y_0) < 8) {
                *(float4 *)(Apad_shared4 + (((((int)threadIdx_y_0) * 128) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4 *)(A0 + ((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_0) * 128)) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
        }
        if (((int)threadIdx_y_0) < 8) {
            *(float4 *)(W_shared5 + ((((int)threadIdx_y_0) * 64) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(W2 + (((rc_outer * 512) + (((int)threadIdx_y_0) * 64)) + (((int)threadIdx_x_0) * 4)));
        }
        asm ("bar.sync 1,256;");
        ;
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
            for (int ax3 = 0; ax3 < 4; ++ax3) {
                Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3)];
                Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3) + 64)];
            }
            for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 64) + (((int)threadIdx_y_0) * 4)) + ax3_1)];
            }
            for (int ff_c = 0; ff_c < 4; ++ff_c) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                    B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                    B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 16;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 16;
    unsigned int blockDim_y_1 = 8;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 16 % 8;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float B_local11[64];
    static float Apad_shared12[512] __attribute__((shared));
    static float W_shared13[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    *(float4 *)(Apad_shared12 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? *(float4 *)(A8 + ((((((((ry * 229376) + (((int)blockIdx.z) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    *(float4 *)(W_shared13 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4)));
                }
                asm ("bar.sync 2,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3) + 32)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1) + 32)];
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
            B9[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2048)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 32)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2080)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(384, 0) void conv2d_small_conv2d_large_fused_kernel_hfuse_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float B_local3[64];
    static float Apad_shared4[1024] __attribute__((shared));
    static float W_shared5[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        asm ("bar.sync 1,128;");
        ;
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
            if (((int)threadIdx_y_0) < 8) {
                *(float4 *)(Apad_shared4 + (((((int)threadIdx_y_0) * 128) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4 *)(A0 + ((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_0) * 128)) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
        }
        if (((int)threadIdx_y_0) < 8) {
            *(float4 *)(W_shared5 + ((((int)threadIdx_y_0) * 64) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(W2 + (((rc_outer * 512) + (((int)threadIdx_y_0) * 64)) + (((int)threadIdx_x_0) * 4)));
        }
        asm ("bar.sync 1,128;");
        ;
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
            for (int ax3 = 0; ax3 < 4; ++ax3) {
                Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3)];
                Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3) + 64)];
            }
            for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 64) + (((int)threadIdx_y_0) * 4)) + ax3_1)];
            }
            for (int ff_c = 0; ff_c < 4; ++ff_c) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                    B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                    B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 32;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 32;
    unsigned int blockDim_y_1 = 8;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 32 % 8;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 256;
    float B_local11[64];
    static float Apad_shared12[512] __attribute__((shared));
    static float W_shared13[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    *(float4 *)(Apad_shared12 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? *(float4 *)(A8 + ((((((((ry * 229376) + (((int)blockIdx.z) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    *(float4 *)(W_shared13 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4)));
                }
                asm ("bar.sync 2,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3) + 32)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1) + 32)];
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
            B9[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2048)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 32)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2080)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(320, 0) void conv2d_small_conv2d_large_fused_kernel_hfuse_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256) && (blockIdx.z < 3364)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[1024] __attribute__((shared));
    static float W_shared5[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        asm ("bar.sync 1,256;");
        ;
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
            if (((int)threadIdx_y_0) < 8) {
                *(float4 *)(Apad_shared4 + (((((int)threadIdx_y_0) * 128) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4 *)(A0 + ((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_0) * 128)) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
        }
        if (((int)threadIdx_y_0) < 8) {
            *(float4 *)(W_shared5 + ((((int)threadIdx_y_0) * 64) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(W2 + (((rc_outer * 512) + (((int)threadIdx_y_0) * 64)) + (((int)threadIdx_x_0) * 4)));
        }
        asm ("bar.sync 1,256;");
        ;
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
            for (int ax3 = 0; ax3 < 4; ++ax3) {
                Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3)];
                Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3) + 64)];
            }
            for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 64) + (((int)threadIdx_y_0) * 4)) + ax3_1)];
            }
            for (int ff_c = 0; ff_c < 4; ++ff_c) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                    B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                    B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 320) && (blockIdx.z < 3136)){
    unsigned int blockDim_x_1 = 8;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 8;
    unsigned int blockDim_y_1 = 8;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 8 % 8;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 64;
    float B_local11[64];
    static float Apad_shared12[512] __attribute__((shared));
    static float W_shared13[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,64;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    *(float4 *)(Apad_shared12 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? *(float4 *)(A8 + ((((((((ry * 229376) + (((int)blockIdx.z) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    *(float4 *)(W_shared13 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4)));
                }
                asm ("bar.sync 2,64;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3) + 32)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1) + 32)];
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
            B9[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2048)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 32)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2080)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(384, 5) void conv2d_small_conv2d_large_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float B_local3[64];
    static float Apad_shared4[1024] __attribute__((shared));
    static float W_shared5[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        asm ("bar.sync 1,128;");
        ;
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
            if (((int)threadIdx_y_0) < 8) {
                *(float4 *)(Apad_shared4 + (((((int)threadIdx_y_0) * 128) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4 *)(A0 + ((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_0) * 128)) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
        }
        if (((int)threadIdx_y_0) < 8) {
            *(float4 *)(W_shared5 + ((((int)threadIdx_y_0) * 64) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(W2 + (((rc_outer * 512) + (((int)threadIdx_y_0) * 64)) + (((int)threadIdx_x_0) * 4)));
        }
        asm ("bar.sync 1,128;");
        ;
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
            for (int ax3 = 0; ax3 < 4; ++ax3) {
                Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3)];
                Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3) + 64)];
            }
            for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 64) + (((int)threadIdx_y_0) * 4)) + ax3_1)];
            }
            for (int ff_c = 0; ff_c < 4; ++ff_c) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                    B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                    B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 32;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 32;
    unsigned int blockDim_y_1 = 8;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 32 % 8;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 256;
    float B_local11[64];
    static float Apad_shared12[512] __attribute__((shared));
    static float W_shared13[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,256;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    *(float4 *)(Apad_shared12 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? *(float4 *)(A8 + ((((((((ry * 229376) + (((int)blockIdx.z) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    *(float4 *)(W_shared13 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4)));
                }
                asm ("bar.sync 2,256;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3) + 32)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1) + 32)];
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
            B9[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2048)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 32)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2080)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
 __global__ __launch_bounds__(384, 5) void conv2d_small_conv2d_large_fused_kernel_hfuse_lb_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    float B_local3[64];
    static float Apad_shared4[1024] __attribute__((shared));
    static float W_shared5[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        asm ("bar.sync 1,256;");
        ;
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
            if (((int)threadIdx_y_0) < 8) {
                *(float4 *)(Apad_shared4 + (((((int)threadIdx_y_0) * 128) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4 *)(A0 + ((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_0) * 128)) + (((int)threadIdx_x_0) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
            }
        }
        if (((int)threadIdx_y_0) < 8) {
            *(float4 *)(W_shared5 + ((((int)threadIdx_y_0) * 64) + (((int)threadIdx_x_0) * 4))) = *(float4 *)(W2 + (((rc_outer * 512) + (((int)threadIdx_y_0) * 64)) + (((int)threadIdx_x_0) * 4)));
        }
        asm ("bar.sync 1,256;");
        ;
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
            for (int ax3 = 0; ax3 < 4; ++ax3) {
                Apad_shared_local6[ax3] = Apad_shared4[(((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3)];
                Apad_shared_local6[(ax3 + 4)] = Apad_shared4[((((rc_inner * 128) + (((int)threadIdx_x_0) * 4)) + ax3) + 64)];
            }
            for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                W_shared_local7[ax3_1] = W_shared5[(((rc_inner * 64) + (((int)threadIdx_y_0) * 4)) + ax3_1)];
            }
            for (int ff_c = 0; ff_c < 4; ++ff_c) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                    B_local3[((ff_c * 4) + nn_c)] = (B_local3[((ff_c * 4) + nn_c)] + (Apad_shared_local6[nn_c] * W_shared_local7[ff_c]));
                    B_local3[(((ff_c * 4) + nn_c) + 16)] = (B_local3[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local6[(nn_c + 4)] * W_shared_local7[ff_c]));
                }
            }
        }
    }
    for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
            B1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner)] = B_local3[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx_y_0) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_0) * 4)) + nn_inner_inner_inner) + 64)] = B_local3[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 16;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 16;
    unsigned int blockDim_y_1 = 8;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 16 % 8;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128;
    float B_local11[64];
    static float Apad_shared12[512] __attribute__((shared));
    static float W_shared13[512] __attribute__((shared));
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
    for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
        for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
                asm ("bar.sync 2,128;");
                ;
                for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                    *(float4 *)(Apad_shared12 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? *(float4 *)(A8 + ((((((((ry * 229376) + (((int)blockIdx.z) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                }
                for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                    *(float4 *)(W_shared13 + (((((int)threadIdx_y_1) * 64) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(W10 + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_1) * 64)) + (((int)threadIdx_x_1) * 8)) + (ax3_inner_outer_1 * 4)));
                }
                asm ("bar.sync 2,128;");
                ;
                for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                    for (int ax3 = 0; ax3 < 4; ++ax3) {
                        Apad_shared_local14[ax3] = Apad_shared12[(((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3)];
                        Apad_shared_local14[(ax3 + 4)] = Apad_shared12[((((rc_inner * 64) + (((int)threadIdx_x_1) * 4)) + ax3) + 32)];
                    }
                    for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                        W_shared_local15[ax3_1] = W_shared13[(((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1)];
                        W_shared_local15[(ax3_1 + 4)] = W_shared13[((((rc_inner * 64) + (((int)threadIdx_y_1) * 4)) + ax3_1) + 32)];
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
            B9[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner)] = B_local11[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2048)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 32)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
            B9[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx_y_1) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_1) * 4)) + nn_inner_inner_inner) + 2080)] = B_local11[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
        }
    }
}
}
