#!/usr/bin/python3

import os, sys
import argparse
import logging
import yaml
import subprocess
#-----------------------------------------------------------------------------------------------
# Settings
metrics_trial = 30
exec_trial    = 100
#-----------------------------------------------------------------------------------------------
def get_valid_commands(infoYAML, benchmark_path):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])
    test_methology = ["kernel1", "kernel2", "parallel", "hfuse", "bfuse"]

    valid_commands = []

    benchmark_path = os.path.abspath(benchmark_path)
    common_command = [benchmark_path, "-v"]

    for idx, test_name in enumerate(test_methology):
        if idx == 0:
            for kidx in range(kernel1_size):
                command = common_command + [str(idx), str(kidx), "0"]
                valid_commands.append(command)
        if idx == 1:
            for kidx in range(kernel2_size):
                command = common_command + [str(idx), "0", str(kidx)]
                valid_commands.append(command)
        if idx == 2 or idx == 3 or idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    command = common_command + [str(idx), str(kidx1), str(kidx2)]
                    valid_commands.append(command)
    
    return valid_commands
#-----------------------------------------------------------------------------------------------
def get_metrics_commands(infoYAML, benchmark_path, metrics_path):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])
    test_methology = ["kernel1", "kernel2"]
    test_metrics   = ["stall_constant_memory_dependency",
                      "stall_exec_dependency",
                      "stall_inst_fetch",
                      "stall_memory_dependency",
                      "stall_memory_throttle",
                      "stall_not_selected",
                      "stall_other",
                      "stall_pipe_busy"
                      "stall_sync",
                      "stall_texture"]

    metrics_commands = []
    common_command   = ["nvprof", "--metrics", ",".join(test_metrics), "--csv", "--log-file"]

    for idx, test_name in enumerate(test_methology):
        if idx == 0:
            for kidx in range(kernel1_size):
                file_path = os.path.join(metrics_path, f"{idx}_{kidx}_0.csv")
                command   = common_command + [file_path]
                command   = command + [benchmark_path, "-n", str(metrics_trial), str(idx), str(kidx), "0"]
                metrics_commands.append(command)
        if idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(metrics_path, f"{idx}_0_{kidx}.csv")
                command   = common_command + [file_path]
                command = command + [benchmark_path, "-n", str(metrics_trial), str(idx), "0", str(kidx)]
                metrics_commands.append(command)
    
    return metrics_commands
#-----------------------------------------------------------------------------------------------
def get_exec_commands(infoYAML, benchmark_path, exec_path):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])
    test_methology = ["kernel1", "kernel2", "parallel", "hfuse", "bfuse"]

    exec_commands  = []
    common_command = ["nvprof", "--print-gpu-trace", "--csv", "--log-file"]

    for idx, test_name in enumerate(test_methology):
        if idx == 0:
            for kidx in range(kernel1_size):
                file_path = os.path.join(exec_path, f"{idx}_{kidx}_0.csv")
                command   = common_command + [file_path]
                command   = command + [benchmark_path, "-n", str(exec_trial), str(idx), str(kidx), "0"]
                exec_commands.append(command)
        if idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(exec_path, f"{idx}_0_{kidx}.csv")
                command   = common_command + [file_path]
                command   = command + [benchmark_path, "-n", str(exec_trial), str(idx), "0", str(kidx)]
                exec_commands.append(command)
        if idx == 2 or idx == 3 or idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(exec_path, f"{idx}_{kidx1}_{kidx2}.csv")
                    command   = common_command + [file_path]
                    command   = command + [benchmark_path, "-n", str(exec_trial), str(idx), str(kidx1), str(kidx2)]
                    exec_commands.append(command)
    
    return exec_commands
#-----------------------------------------------------------------------------------------------
def get_profile_data(infoYAML, benchmark_path, profile_path, valid=False, profile_metrics=False, profile_exec=False):

    # Validation check
    if valid:
        valid_commands = get_valid_commands(infoYAML, benchmark_path)
        for idx, command in enumerate(valid_commands):
            print(f"[{idx+1}/{len(valid_commands)}] Validation check : ", end="")
            try:
                result = subprocess.run(command,
                                        stdout=subprocess.PIPE,
                                        text=True,
                                        check=True,
                                        # shell=True,
                                        # timeout=10,
                                        )
            except subprocess.CalledProcessError as e:
                print("INVALID")
                exit(1)
            else:
                print("VALID")
    
    # Profile metrics
    if profile_metrics:
        metrics_path = os.path.join(profile_path, "metrics")
        os.mkdir(metrics_path)

        metrics_commands = get_metrics_commands(infoYAML, benchmark_path, metrics_path)
        for idx, command in enumerate(metrics_commands):
            print(f"[{idx+1}/{len(metrics_commands)}] Profile metrics...")
            try:
                result = subprocess.run(command,
                                        stdout=subprocess.PIPE,
                                        text=True,
                                        check=True,
                                        # timeout=10,
                                        )
            except subprocess.CalledProcessError as e:
                logging.error("Error occurs while profiling metrics.")
                exit(1)

    # Profile exectuion
    if profile_exec:
        exec_path = os.path.join(profile_path, "exec")
        os.mkdir(exec_path)

        exec_commands = get_exec_commands(infoYAML, benchmark_path, exec_path)
        for idx, command in enumerate(exec_commands):
            print(f"[{idx+1}/{len(exec_commands)}] Profile execution...")
            try:
                result = subprocess.run(command,
                                        stdout=subprocess.PIPE,
                                        text=True,
                                        check=True,
                                        # timeout=10,
                                        )
            except subprocess.CalledProcessError as e:
                logging.error("Error occurs while profiling executions.")
                exit(1)
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true", default=False, dest="valid",
                        help="options for validation check")
    parser.add_argument("-m", action="store_true", default=False, dest="metrics",
                        help="options for profiling metrics")
    parser.add_argument("-e", action="store_true", default=False, dest="exec",
                        help="options for profiling executions")
    parser.add_argument("file", action="store", help="path of generated test-suite")

    # Get arguments
    args   = parser.parse_args()
    output = args.file
    valid  = args.valid

    profile_metrics = args.metrics
    profile_exec    = args.exec
    
    output = os.path.abspath(output)

    # Check the neccessary directories/files exist
    if not os.path.exists(output):
        logging.error("Given config path \"%s\" doesn't exist." % output)
        exit(1)

    benchmark_path = os.path.join(output, "benchmark")
    if not os.path.exists(benchmark_path):
        logging.error("Given config path \"%s\" doesn't exist." % benchmark_path)
        exit(1)

    # Parse YAML files
    config_path = os.path.join(output, "config")
    info_path   = os.path.join(config_path, "info.yaml")

    with open(info_path) as f:
        yaml_info = yaml.safe_load(f)

    # Create profile data directory
    profile_path = os.path.join(output, "profile")
    if os.path.exists(profile_path):
        logging.error("\"%s\" alreay exists." % profile_path)
        exit(1)

    os.mkdir(profile_path)

    # Get profile data with benchmark
    get_profile_data(yaml_info, benchmark_path, profile_path, valid, profile_metrics, profile_exec)
#-----------------------------------------------------------------------------------------------