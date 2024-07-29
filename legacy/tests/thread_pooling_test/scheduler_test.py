
import re

with open("results.txt") as f:
    lines = f.readlines()

    # A6000
    sm_list = [0] * 84

    for line in lines:
        block = re.search(r"Block: \d*", line)
        sm    = re.search(r"SM: \d*", line)

        block_id = block.group().split(" ")[1]
        sm_id    = sm.group().split(" ")[1]

        sm_list[int(sm_id)] += 1

    print(sm_list)

