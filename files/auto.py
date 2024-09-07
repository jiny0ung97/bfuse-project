#!/usr/bin/python3

# file_name = "bert-base-uncased.txt"
# file_name = "gpt2.txt"
# file_name = "Meta-Llama-3-8B.txt"
file_name = "resnet-18.txt"

with open("result.txt", "w") as wf:
    with open(file_name, "r") as rf:
        lines = rf.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                wf.write(line)
                wf.write("\n")
            if "nn.batch_matmul" in line or "nn.dense" in line:
                if idx > 1:
                    wf.write(lines[idx-2])
                if idx > 0:
                    wf.write(lines[idx-1])
                wf.write(line)
                wf.write("\n")
            elif "nn.conv2d" in line:
                if idx > 0:
                    wf.write(lines[idx-1])
                wf.write(line)
                wf.write("\n")