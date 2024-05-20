#-----------------------------------------------------------------------------------------------
def get_utils_h():
    utils_h = \
"""
#pragma once

double get_current_time();

void alloc_tensor(float **m, int N, int C, int H, int W);

void rand_tensor(float *m, int N, int C, int H, int W);

void zero_tensor(float *m, int N, int C, int H, int W);

void print_tensor(float *m, int N, int C, int H, int W);

bool check_matrix(float *O, float *O_ans, int N, int C, int H, int W);
"""

    return utils_h
#-----------------------------------------------------------------------------------------------