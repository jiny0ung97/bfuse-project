#!/usr/bin/python3

import re
import sys
import math

def check(idx):
    # if idx < 1924096 and idx % 3758 >= 0 and idx % 3758 < 128:
    #     kernel_id = 0
    # elif idx < 1924096 and idx % 3758 >= 128 and idx % 3758 < 3758:
    #     kernel_id = 1
    # else:
    #     print("error")
    #     exit()

    if idx < 172032 and idx % 336 >= 0 and idx % 336 < 128:
        kernel_id = 0
    elif idx < 172032 and idx % 336 >= 128 and idx % 336 < 336:
        kernel_id = 1
    else:
        print(f"error : {idx}")
        exit()

    return kernel_id

# Colors
RED   = "\033[31m" # 0
BLUE  = "\033[34m" # 1
RESET = "\033[0m" # reset

max_block  = 0
max_sm     = 0
min_block  = math.inf
min_sm     = math.inf
max_length = 70
max_iter   = 4000
sm_num     = 82 # RTX 3090
file       = sys.argv[1]

with open(file) as f:
    lines = f.readlines()

    # Initialize lists
    sm_lists = []
    for idx1 in range(0, max_length):
        tmp_list = []
        for idx2 in range(0, sm_num):
            tmp_list.append(-1)
        sm_lists.append(tmp_list)

    for iter, line in enumerate(lines):
        block  = re.search(r"Block: \d*", line)
        sm     = re.search(r"SM: \d*", line)

        block_id  = int(block.group().split(" ")[1])
        sm_id     = int(sm.group().split(" ")[1])

        if block_id > max_block:
            max_block = block_id
        if block_id < min_block:
            min_block = block_id
        if sm_id > max_sm:
            max_sm = sm_id
        if sm_id < min_sm:
            min_sm = sm_id

        for idx in range(0, max_length):
            if sm_lists[idx][sm_id] == -1:
                sm_lists[idx][sm_id] = block_id
                break
            elif sm_lists[idx][sm_id] > block_id:
                tmp = sm_lists[idx][sm_id]
                sm_lists[idx][sm_id] = block_id
                block_id = tmp

        if iter == max_iter:
            break

    # Statistics
    max_rate = 0
    min_rate = math.inf
    max_rate_0 = 0
    max_rate_1 = 0
    min_rate_0 = 0
    min_rate_1 = 0
    for sm_idx in range(0, max_sm):
        rate_0 = 0
        rate_1 = 0
        for idx in range(0, max_length):
            if sm_lists[idx][sm_idx] == -1:
                continue
            if check(sm_lists[idx][sm_idx]) == 0:
                rate_0 += 1
            elif check(sm_lists[idx][sm_idx]) == 1:
                rate_1 += 1

        if abs(rate_0 - rate_1) > abs(max_rate):
            max_rate_0 = rate_0
            max_rate_1 = rate_1
        if abs(rate_0 - rate_1) < abs(min_rate):
            min_rate_0 = rate_0
            min_rate_1 = rate_1

    # Print Kernel configurations
    print(f"Blocks: {min_block} ~ {max_block}")
    print(f"SMs:    {min_sm} ~ {max_sm}")
    print(f"Max Diff.: {max_rate_0 / (max_rate_0 + max_rate_1)} : {max_rate_1 / (max_rate_0 + max_rate_1)}")
    print(f"Min Diff.: {min_rate_0 / (min_rate_0 + min_rate_1)} : {min_rate_1 / (min_rate_0 + min_rate_1)}")

    # Print allocated blocks
    for list in sm_lists:
        for e in list:
            if e == -1:
                continue
            if check(e) == 0:
                color     = RED
                kernel_id = 0
            elif check(e) == 1:
                color     = BLUE
                kernel_id = 1
            
            # print(color + str(kernel_id) + RESET, end="")
            print(color + str(e) + RESET, end=" ")
        print("", end="\n")