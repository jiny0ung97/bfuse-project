#!/usr/bin/python3

import subprocess
import logging

start_idx = 0
end_idx   = 5
for idx in range(start_idx, end_idx+1):
    command1 = ["sed", "-i", f"s/int M = {2**(idx-1)};/int M = {2**idx};/g",
                "../../horizontal-fuser/src/Algorithms.cc"]
    command2 = ["cmake", "--build", "../../horizontal-fuser/build/"]
    command3 = ["../../scripts/quick_start.py", "-o", f"M_diff/case-{2**idx}", "../../test-utils/examples/single_bgemm_conv2d.yaml"]

    try:
        if idx != 0:
            print(f"({idx}/{end_idx}) RUN: " + " ".join(command1))
            result1 = subprocess.run(command1,
                                        # stdout=subprocess.PIPE,
                                        text=True,
                                        check=True,
                                        # timeout=10,
                                        )
        print(f"({idx}/{end_idx}) RUN: " + " ".join(command2))
        result2 = subprocess.run(command2,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
        print(f"({idx}/{end_idx}) RUN: " + " ".join(command3))
        result3 = subprocess.run(command3,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while make.")
        exit(1)

command = ["sed", "-i", f"s/int M = {2**(end_idx)};/int M = 1;/g",
            "../../horizontal-fuser/src/Algorithms.cc"]
try:
    print(f"(=/=) RUN: " + " ".join(command))
    result1 = subprocess.run(command,
                                # stdout=subprocess.PIPE,
                                text=True,
                                check=True,
                                # timeout=10,
                                )
except subprocess.CalledProcessError as e:
    logging.error(f"Error occurs while make.")
    exit(1)

start_idx = 0
end_idx   = 8
for idx in range(start_idx, end_idx+1):
    command1 = ["sed", "-i", f"s/int N = {2**(idx-1)};/int N = {2**idx};/g",
                "../../horizontal-fuser/src/Algorithms.cc"]
    command2 = ["cmake", "--build", "../../horizontal-fuser/build/"]
    command3 = ["../../scripts/quick_start.py", "-o", f"N_diff/case-{2**idx}", "../../test-utils/examples/single_bgemm_conv2d.yaml"]

    try:
        if idx != 0:
            print(f"({idx}/{end_idx}) RUN: " + " ".join(command1))
            result1 = subprocess.run(command1,
                                        # stdout=subprocess.PIPE,
                                        text=True,
                                        check=True,
                                        # timeout=10,
                                        )
        print(f"({idx}/{end_idx}) RUN: " + " ".join(command2))
        result2 = subprocess.run(command2,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
        print(f"({idx}/{end_idx}) RUN: " + " ".join(command3))
        result3 = subprocess.run(command3,
                                    # stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    # timeout=10,
                                    )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while make.")
        exit(1)