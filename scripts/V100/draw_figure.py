#!/usr/bin/python3

import os, sys
import argparse
import logging
import yaml, csv
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#-----------------------------------------------------------------------------------------------
# Settings
warmup_trials  = 3
metrics_trials = 10
exec_trials    = 50
#-----------------------------------------------------------------------------------------------
def parse_csv(csv_path):
    nvvp_list = []
    with open(csv_path, "r") as f:
        lines = csv.reader(f)
        for idx, line in enumerate(lines):
            if idx < 3:
                # ==2119== NVPROF is profiling process 2119, command: ./fusion_bench -n 10 -t 0 0
                # ==2119== Profiling application: ./fusion_bench -n 10 -t 0 0
                # ==2119== Profiling result:
                continue
            elif idx == 3:
                # "Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","SrcMemType","DstMemType","Device","Context","Stream","Name","Correlation_ID"
                types = line
                continue
            elif idx ==4:
                # ms,ms,,,,,,,,KB,B,MB,GB/s,,,,,,,
                units = line
                continue
            if not line:
                continue

            # Parse CSV file
            row_dict = {}
            for i, data in enumerate(line):
                type = types[i]

                if type == "Start" or type == "Duration":
                    unit = units[i]
                    if unit == "s":
                        row_dict[type] = float(data) * 1000
                    elif unit == "ms":
                        row_dict[type] = float(data)
                    elif unit == "us":
                        row_dict[type] = float(data) / 1000
                    else:
                        print("Error: unknown unit")
                else:
                    row_dict[type] = data

            nvvp_list.append(row_dict)
    
    return nvvp_list, types
#-----------------------------------------------------------------------------------------------
def get_single_statistics(datas, kernel_name):
    count   = 0
    results = []
    for data in datas:
        name = data.get("Name")
        if name != kernel_name:
            continue

        count = count + 1
        if count <= warmup_trials:
            continue

        duration = float(data.get("Duration"))
        results.append(duration)

    result_min  = np.min(results)
    result_max  = np.max(results)
    result_mean = np.mean(results)
    result_std  = np.std(results)

    return result_min, result_max, result_mean, result_std
#-----------------------------------------------------------------------------------------------
def get_parallel_statistics(datas, kernel_names):
    count   = 0
    results = []

    if len(kernel_names) != 2:
        loggging.error("Number of kernel_names must be 2.")
        exit(1)

    for data in datas:
        name = data.get("Name")
        if name not in kernel_names:
            continue

        count = count + 1
        if count <= warmup_trials * len(kernel_names):
            continue

        if count % 2 == 1:
            start1    = float(data.get("Start"))
            duration1 = float(data.get("Duration"))
        elif count % 2 == 0:
            start2    = float(data.get("Start"))
            duration2 = float(data.get("Duration"))

            if start1 + duration1 > start2 + duration2:
                exec_time = duration1
            else:
                exec_time = start2 - start1 + duration2

            results.append(exec_time)

    result_min  = np.min(results)
    result_max  = np.max(results)
    result_mean = np.mean(results)
    result_std  = np.std(results)

    return result_min, result_max, result_mean, result_std
#-----------------------------------------------------------------------------------------------
def preprocess_datas(infoYAML, profile_path):
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

    kernel1_datas  = []
    kernel2_datas  = []
    parallel_datas = []
    hfuse_datas    = []
    bfuse_datas    = []

    for idx, test_name in enumerate(test_methology):
        if idx == 0:
            for kidx in range(kernel1_size):
                file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx}_0.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[0]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel1_datas.append(data_mean)
        elif idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(profile_path, "exec", f"{idx}_0_{kidx}.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[1]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel2_datas.append(data_mean)
        elif idx == 2:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx1}_{kidx2}.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_parallel_statistics(data, [kname1, kname2])
                    parallel_datas.append(data_mean)
        elif idx == 3:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx1}_{kidx2}.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_hfuse")
                    hfuse_datas.append(data_mean)
        elif idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx1}_{kidx2}.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_bfuse")
                    bfuse_datas.append(data_mean)

    return kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas
#-----------------------------------------------------------------------------------------------
def draw_exec_graph(infoYAML, profile_path, output_path):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(1)

    kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas = preprocess_datas(infoYAML, profile_path)
    
    # Preprocess datas
    kernel1 = fusion_sets[0]["Set"]
    kernel2 = fusion_sets[1]["Set"]

    materials = []
    parallel  = []
    hfuse     = []
    bfuse     = []

    for bi in range(len(kernel1)):
        for ci in range(len(kernel2)):
            materials.append("%s x %s" % (kernel1[bi], kernel2[ci]))

            serial_exec   = kernel1_datas[bi] + kernel2_datas[ci]
            parallel_exec = serial_exec / parallel_datas[bi * len(kernel2) + ci]
            hfuse_exec    = serial_exec / hfuse_datas[bi * len(kernel2) + ci]
            bfuse_exec    = serial_exec / bfuse_datas[bi * len(kernel2) + ci]
            
            parallel.append(parallel_exec)
            hfuse.append(hfuse_exec)
            bfuse.append(bfuse_exec)

    print(f"parallel | max: {np.max(parallel)}, min: {np.min(parallel)}, avg: {np.mean(parallel)}")
    print(f"hfuse    | max: {np.max(hfuse)}, min: {np.min(hfuse)}, avg: {np.mean(hfuse)}")
    print(f"bfuse    | max: {np.max(bfuse)}, min: {np.min(bfuse)}, avg: {np.mean(bfuse)}")

    # Settings
    if len(kernel1) > 1:
        f, axes = plt.subplots(nrows=len(kernel1), ncols=1, sharex=False, sharey=True)
    else:
        # temp declaration to avoid erros
        f, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True)

    # Draw
    for bi in range(len(kernel1)):
        index  = np.arange(len(kernel2))
        axes_m = materials[bi * len(kernel2): (bi + 1) * len(kernel2)]
        axes_p = parallel[bi * len(kernel2): (bi + 1) * len(kernel2)]
        axes_h = hfuse[bi * len(kernel2): (bi + 1) * len(kernel2)]
        axes_b = bfuse[bi * len(kernel2): (bi + 1) * len(kernel2)]

        axes[bi].bar(index - 0.2, axes_p, width=0.2, label='parallel')
        axes[bi].bar(index      , axes_h, width=0.2, label='hfuse')
        axes[bi].bar(index + 0.2, axes_b, width=0.2, label='bfuse')

        # Draw baseline
        axes[bi].axhline(y=1, color="r", linewidth=0.5, linestyle="--")

        # Set ticks
        # axes[bi].set_xticks(index, axes_m, fontsize=3)
        axes[bi].set_xticks(index, [], fontsize=3)

        # Set limitation of y axis
        axes[bi].set_ylim(0, math.ceil(max(hfuse + bfuse)))

        # Set major & minor yticks
        axes[bi].yaxis.set_major_locator(ticker.MultipleLocator(1))
        axes[bi].yaxis.set_major_formatter("{x:.0f}")
        axes[bi].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        axes[bi].yaxis.set_minor_formatter("{x:.0f}")

        # Set major & minor yticklabels
        # yticks = axes[bi].get_yticks()
        # axes[bi].set_yticklabels(yticks, fontdict={"fontsize": 3})
        axes[bi].set_yticklabels([], fontdict={"fontsize": 3})
        # yticks = axes[bi].get_yticks(minor=True)
        # axes[bi].set_yticklabels(yticks, fontdict={"fontsize": 2}, minor=True)
        axes[bi].set_yticklabels([], fontdict={"fontsize": 3}, minor=True)

        # axes[bi].set_ylabel('Speed up', fontsize=3)
        # axes[bi].legend(ncol=1, loc="upper left", fontsize=2)

    # Save figure
    f.tight_layout()

    exec_figure = os.path.join(output_path, "exec_figure.png")
    plt.savefig(exec_figure, dpi=500)
#-----------------------------------------------------------------------------------------------
def draw_metrics_graph(infoYAML, profile_path, output_path):
    pass
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", action="store", help="path of generated test-suite")
    parser.add_argument("-o", action="store", default=".", dest="output",
                        help="output file name of figure")

    # Get arguments
    args        = parser.parse_args()
    file_path   = args.file
    output_path = args.output
    
    if not os.path.exists(file_path):
        logging.error("Given config path \"%s\" doesn't exist." % file_path)
        exit(1)

    # Parse YAML files
    config_path  = os.path.join(file_path, "config")
    profile_path = os.path.join(file_path, "profile")
    info_path    = os.path.join(config_path, "info.yaml")

    with open(info_path) as f:
        yaml_info = yaml.safe_load(f)

    draw_exec_graph(yaml_info, profile_path, output_path)
    draw_metrics_graph(yaml_info, profile_path, output_path)
#-----------------------------------------------------------------------------------------------