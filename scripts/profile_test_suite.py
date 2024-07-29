#!/usr/bin/python3

import os, sys
import argparse
import logging
import yaml
import subprocess
#-----------------------------------------------------------------------------------------------
# Settings
metrics_trials = 10
exec_trials    = 30
#-----------------------------------------------------------------------------------------------
def get_valid_commands(infoYAML, benchmark_path):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(1)

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
        exit(1)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])
    test_methology = ["kernel1", "kernel2"]

    metrics_commands = []
    common_command   = ["ncu", "--set", "full", "-o"]

    for idx, test_name in enumerate(test_methology):
        if idx == 0:
            for kidx in range(kernel1_size):
                file_path = os.path.join(metrics_path, f"{idx}_{kidx}_0")
                command   = common_command + [file_path]
                command   = command + [benchmark_path, "-n", str(metrics_trials), str(idx), str(kidx), "0"]
                metrics_commands.append(command)
        if idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(metrics_path, f"{idx}_0_{kidx}")
                command   = common_command + [file_path]
                command = command + [benchmark_path, "-n", str(metrics_trials), str(idx), "0", str(kidx)]
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
        exit(1)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])
    test_methology = ["kernel1", "kernel2", "parallel", "hfuse", "bfuse"]

    exec_commands  = []

    # Profile timeline (.nsys-rep) & Report into file (.csv)
    common_command1 = ["nsys", "nvprof", "--print-gpu-trace", "-o"]
    common_command2 = ["nsys", "stats", "--force-export=true",
                       "--report", "cuda_gpu_trace", "--format", "csv",
                       "-o"]
    common_command3 = ["rm", "-Rf"]
    for idx, test_name in enumerate(test_methology):
        if idx == 0:
            for kidx in range(kernel1_size):
                file_path = os.path.join(exec_path, f"{idx}_{kidx}_0")
                command1  = common_command1 + [file_path]
                command1  = command1 + [benchmark_path, "-n", str(exec_trials), str(idx), str(kidx), "0"]
                command2  = common_command2 + [file_path]
                command2  = command2 + [file_path + ".nsys-rep"]
                command3  = common_command3 + [file_path + ".nsys-rep", file_path + ".sqlite"]
                exec_commands.append([command1, command2, command3])
        if idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(exec_path, f"{idx}_0_{kidx}")
                command1  = common_command1 + [file_path]
                command1  = command1 + [benchmark_path, "-n", str(exec_trials), str(idx), "0", str(kidx)]
                command2  = common_command2 + [file_path]
                command2  = command2 + [file_path + ".nsys-rep"]
                command3  = common_command3 + [file_path + ".nsys-rep", file_path + ".sqlite"]
                exec_commands.append([command1, command2, command3])
        if idx == 2 or idx == 3 or idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(exec_path, f"{idx}_{kidx1}_{kidx2}")
                    command1  = common_command1 + [file_path]
                    command1  = command1 + [benchmark_path, "-n", str(exec_trials), str(idx), str(kidx1), str(kidx2)]
                    command2  = common_command2 + [file_path]
                    command2  = command2 + [file_path + ".nsys-rep"]
                    command3  = common_command3 + [file_path + ".nsys-rep", file_path + ".sqlite"]
                    exec_commands.append([command1, command2, command3])

    return exec_commands
#-----------------------------------------------------------------------------------------------
def get_profile_data(infoYAML, benchmark_path, profile_path, valid=False, profile_metrics=False, profile_exec=False):

    # Validation check
    if valid:
        valid_commands = get_valid_commands(infoYAML, benchmark_path)
        for idx, command in enumerate(valid_commands):
            print(f"({idx+1}/{len(valid_commands)}) Validation check : ", end="")
            try:
                result = subprocess.run(command,
                                        # stdout=subprocess.PIPE,
                                        text=True,
                                        check=True,
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
            print(f"({idx+1}/{len(metrics_commands)}) Profile metrics...")
            try:
                result = subprocess.run(command,
                                        # stdout=subprocess.PIPE,
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
        for idx, commands in enumerate(exec_commands):
            print(f"({idx+1}/{len(exec_commands)}) Profile execution...")
            for command in commands:
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
    args      = parser.parse_args()
    file_path = args.file
    valid     = args.valid

    profile_metrics = args.metrics
    profile_exec    = args.exec
    
    output = os.path.abspath(file_path)

    # Check the neccessary directories/files exist
    if not os.path.exists(file_path):
        logging.error("Given config path \"%s\" doesn't exist." % file_path)
        exit(1)

    benchmark_path = os.path.join(file_path, "benchmark")
    if not os.path.exists(benchmark_path):
        logging.error("Given config path \"%s\" doesn't exist." % benchmark_path)
        exit(1)

    # Parse YAML files
    config_path = os.path.join(file_path, "config")
    info_path   = os.path.join(config_path, "info.yaml")

    with open(info_path) as f:
        yaml_info = yaml.safe_load(f)

    # Create profile data directory
    profile_path = os.path.join(file_path, "profile")
    if os.path.exists(profile_path):
        logging.error("\"%s\" alreay exists." % profile_path)
        exit(1)

    os.mkdir(profile_path)

    # Get profile data with benchmark
    get_profile_data(yaml_info, benchmark_path, profile_path, valid, profile_metrics, profile_exec)
#-----------------------------------------------------------------------------------------------