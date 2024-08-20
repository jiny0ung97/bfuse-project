#!/usr/bin/python3

import subprocess
import logging

max_idx = 29
for idx in range(3, max_idx+1):
    command1 = ["sed", "-i", f"s/- conv2d_{idx-1}/- conv2d_{idx}/g",
                "../test-utils/examples/inception_v3.yaml"]
    command2 = ["ncu", "--set", "full", "-o", f"temp/conv2d_{idx}",
                "../scripts/configure_kernels.py", "-e", "../test-utils/examples/inception_v3.yaml"]
    command3 = ["rm", "-rf", "test-suite/"]

    try:
        print(f"({idx}/{max_idx}) RUN: " + " ".join(command1))
        result1 = subprocess.run(command1,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
        print(f"({idx}/{max_idx}) RUN: " + " ".join(command2))
        result2 = subprocess.run(command2,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
        print(f"({idx}/{max_idx}) RUN: " + " ".join(command3))
        result3 = subprocess.run(command3,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while make.")
        exit(1)