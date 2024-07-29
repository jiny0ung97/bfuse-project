#!/usr/bin/python3

import re
import sys

# Colors
RED   = "\033[31m" # 0
BLUE  = "\033[34m" # 1
RESET = "\033[0m" # reset

max_block  = 0
max_sm     = 0
max_length = 70
max_iter   = 10000
sm_num     = 82 # RTX 3090
file       = sys.argv[1]

# Main
with open(file) as f:
    lines = f.readlines()

    # Initialize lists
    sm_lists = []
    for idx1 in range(0, max_length):
        tmp_list = []
        for idx2 in range(0, sm_num):
            tmp_list.append(" ")
        sm_lists.append(tmp_list)

    for iter, line in enumerate(lines):
        block  = re.search(r"Block: \d*", line)
        sm     = re.search(r"SM: \d*", line)
        kernel = re.search(r"Kernel: \d*", line)

        block_id  = int(block.group().split(" ")[1])
        sm_id     = int(sm.group().split(" ")[1])
        kernel_id = int(kernel.group().split(" ")[1])

        if block_id > max_block:
            max_block = block_id
        if sm_id > max_sm:
            max_sm = sm_id

        if kernel_id == 0:
            color = RED
        elif kernel_id == 1:
            color = BLUE
        else:
            print("error: Undefined kernel id")
            exit(1)

        for idx in range(0, max_length):
            if sm_lists[idx][sm_id] == " ":
                # sm_lists[idx][sm_id] = color + str(block_id) + RESET
                sm_lists[idx][sm_id] = color + str(kernel_id) + RESET
                break

        if iter == max_iter:
            break

    # Print Kernel configurations
    print(f"Max Block: {max_block}")
    print(f"Max SM:    {max_sm}")
        
    # Print allocated blocks
    for list in sm_lists:
        for e in list:
            print(e, end="")
        print("", end="\n")

