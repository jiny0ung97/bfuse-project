#!/usr/bin/python3

import os, sys
import argparse
import logging
import yaml, csv
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
#-----------------------------------------------------------------------------------------------
# conv2d_0  - issue rate: 5.4,  stall: 10.52, (MIO, ),  throughput: 86.45/86.45, utilization(LSU): 86.30
# conv2d_1  - issue rate: 3.6,  stall: 21.29, (MIO, ),  throughput: 98.42/98.42, utilization(LSU): 98.43
# conv2d_2  - issue rate: 3.6,  stall: 20.81, (MIO, ),  throughput: 98.23/98.23, utilization(LSU): 98.23
# conv2d_3  - issue rate: 24.9, stall: 54.06, (MIO, ),  throughput: 13.90/73.82, utilization(LSU): 13.90
# conv2d_4  - issue rate: 32.1, stall: 78.99, (MIO, ),  throughput: 15.97/86.38, utilization(LSU): 15.97
# conv2d_5  - issue rate: 9.0,  stall: 17.40, (MIO, ),  throughput: 21.58/82.36, utilization(LSU): 21.60
# conv2d_6  - issue rate: 6.6,  stall: 12.09, (MIO, ),  throughput: 29.02/84.08, utilization(LSU): 29.05
# conv2d_7  - issue rate: 2.6,  stall: 9.60,  (MIO, ),   throughput: 95.98/95.95, utilization(LSU): 96.05
# conv2d_8  - issue rate: 8.5,  stall: 15.89, (Long, ), throughput: 24.87/77.65, utilization(LSU): 24.88
# conv2d_9  - issue rate: 2.5,  stall: 4.95,  (Long, ),  throughput: 59.25/78.40, utilization(LSU): 59.34
# conv2d_10 - issue rate: 8.6,  stall: 12.79, (MIO, ),  throughput: 23.14/72.95, utilization(LSU): 23.15
# conv2d_11 - issue rate: 2.4,  stall: 2.60,  (MIO, ),   throughput: 55.99/91.07, utilization(LSU): 56.17
# conv2d_12 - issue rate: 2.4,  stall: 4.20,  (MIO, ),   throughput: 77.38/85.80, utilization(LSU): 77.55
# conv2d_13 - issue rate: 3.5,  stall: 5.64,  (Long, ),  throughput: 71.12/85.26, utilization(LSU): 71.18
# conv2d_14 - issue rate: 3.1,  stall: 5.47,  (MIO, ),   throughput: 56.90/94.54, utilization(LSU): 57.15
# conv2d_15 - issue rate: 7.0,  stall: 17.02, (MIO, ), throughput: 82.71/86.38, utilization(LSU): 82.72
# conv2d_16 - issue rate: 17.9, stall: 41.63, (MIO, ),  throughput: 31.16/91.49, utilization(LSU): 31.17
# conv2d_17 - issue rate: 20.2, stall: 33.81, (MIO, ),  throughput: 24.55/98.75, utilization(LSU): 24.55
# conv2d_18 - issue rate: 13.6, stall: 19.67, (Long, ), throughput: 36.46/98.56, utilization(LSU): 36.46
# conv2d_19 - issue rate: 30.4, stall: 60.12, (MIO, ),  throughput: 14.01/75.55, utilization(LSU): 14.01
# conv2d_20 - issue rate: 3.2,  stall: 10.30, (Long, ), throughput: 52.83/91.92, utilization(LSU): 52.89
# conv2d_21 - issue rate: 4.0,  stall: 15.84, (MIO, ),  throughput: 58.73/99.03, utilization(LSU): 58.83
# conv2d_22 - issue rate: 2.7,  stall: 3.86,  (Long, ),  throughput: 58.44/81.96, utilization(LSU): 58.55
# conv2d_23 - issue rate: 4.0,  stall: 16.18, (MIO, ),  throughput: 58.76/99.09, utilization(LSU): 58.83
# conv2d_24 - issue rate: 2.9,  stall: 18.33, (MIO, ),  throughput: 95.41/95.41, utilization(LSU): 95.60
# conv2d_25 - issue rate: 2.8,  stall: 4.13,  (Long, ),  throughput: 57.62/80.81, utilization(LSU): 57.78
# conv2d_26 - issue rate: 4.3,  stall: 22.19, (MIO, ),  throughput: 55.38/93.37, utilization(LSU): 55.44
# conv2d_27 - issue rate: 2.8,  stall: 3.84, 3.45(MIO, Long),  throughput: 56.41/79.05, utilization(LSU): 56.50
# conv2d_28 - issue rate: 4.3,  stall: 22.57, 5.31(MIO, Long),  throughput: 55.37/93.35, utilization(LSU): 55.43
# conv2d_29 - issue rate: 2.8,  stall: 3.70, 3.57(Long, MIO),  throughput: 55.82/78.22, utilization(LSU): 55.98
metric_datas = [
    {"issue": 5.4, "stall": [10.52, "MIO"], "throughput": [86.45, 86.45]},
    {"issue": 3.6, "stall": [21.29, "MIO"], "throughput": [98.42, 98.42]},
    {"issue": 3.6, "stall": [20.81, "MIO"], "throughput": [98.23, 98.23]},
    {"issue": 24.9, "stall": [54.06, "MIO"], "throughput": [13.90, 73.82]},
    {"issue": 32.1, "stall": [78.99, "MIO"], "throughput": [15.97, 86.38]},
    {"issue": 9.0, "stall": [17.40, "MIO"], "throughput": [21.58, 82.36]},
    {"issue": 6.6, "stall": [12.09, "MIO"], "throughput": [29.02, 84.08]},
    {"issue": 2.6, "stall": [9.60, "MIO"], "throughput": [95.98, 95.95]},
    {"issue": 8.5, "stall": [15.89, "Long"], "throughput": [24.87, 77.65]},
    {"issue": 2.5, "stall": [4.95, "Long"], "throughput": [59.25, 78.40]},
    {"issue": 8.6, "stall": [12.79, "MIO"], "throughput": [23.14, 72.95]},
    {"issue": 2.4, "stall": [2.60, "MIO"], "throughput": [55.99, 91.07]},
    {"issue": 2.4, "stall": [4.20, "MIO"], "throughput": [77.38, 85.80]},
    {"issue": 3.5, "stall": [5.64, "Long"], "throughput": [71.12, 85.26]},
    {"issue": 3.4, "stall": [5.47, "MIO"], "throughput": [56.90, 94.54]},
    {"issue": 7.0, "stall": [17.02, "MIO"], "throughput": [82.71, 86.38]},
    {"issue": 17.9, "stall": [41.63, "MIO"], "throughput": [31.16, 91.49]},
    {"issue": 20.2, "stall": [33.81, "MIO"], "throughput": [24.55, 98.75]},
    {"issue": 13.6, "stall": [19.67, "Long"], "throughput": [36.46, 98.56]},
    {"issue": 30.4, "stall": [60.12, "MIO"], "throughput": [14.01, 75.55]},
    {"issue": 3.2, "stall": [10.30, "Long"], "throughput": [52.83, 91.92]},
    {"issue": 4.0, "stall": [15.84, "MIO"], "throughput": [58.73, 99.03]},
    {"issue": 2.7, "stall": [3.86, "Long"], "throughput": [58.44, 81.96]},
    {"issue": 4.0, "stall": [16.18, "MIO"], "throughput": [58.76, 99.09]},
    {"issue": 2.9, "stall": [18.33, "MIO"], "throughput": [95.41, 95.41]},
    {"issue": 2.8, "stall": [4.13, "Long"], "throughput": [57.62, 80.81]},
    {"issue": 4.3, "stall": [22.19, "MIO"], "throughput": [55.38, 93.37]},
    {"issue": 2.8, "stall": [3.84, "MIO"], "throughput": [56.41, 79.05]},
    {"issue": 4.3, "stall": [22.57, "MIO"], "throughput": [55.37, 93.35]},
    {"issue": 2.8, "stall": [3.70, "Long"], "throughput": [55.82, 78.22]},
]
#-----------------------------------------------------------------------------------------------
# Settings
warmup_trials  = 3
metrics_trials = 1
exec_trials    = 30
#-----------------------------------------------------------------------------------------------
def parse_csv(csv_path):
    nvvp_list = []
    with open(csv_path, "r") as f:
        lines = csv.reader(f)
        for idx, line in enumerate(lines):
            # if idx < 3:
            #     # ==2119== NVPROF is profiling process 2119, command: ./fusion_bench -n 10 -t 0 0
            #     # ==2119== Profiling application: ./fusion_bench -n 10 -t 0 0
            #     # ==2119== Profiling result:
            #     continue
            if idx == 0:
                # Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name
                types = line
                continue
            # elif idx ==4:
            #     # ms,ms,,,,,,,,KB,B,MB,GB/s,,,,,,,
            #     units = line
            #     continue
            if not line:
                continue

            # Parse CSV file
            row_dict = {}
            for i, data in enumerate(line):
                type = types[i]

                if type.startswith("Start") or type.startswith("Duration"):
                    unit = type.split(" ")[-1].strip("()")
                    type = type.split(" ")[0]
                    if unit == "s":
                        row_dict[type] = float(data) * 1000
                    elif unit == "ms":
                        row_dict[type] = float(data)
                    elif unit == "us":
                        row_dict[type] = float(data) / 1000
                    elif unit == "ns":
                        row_dict[type] = float(data) / 1000000
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
                file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx}_0_cuda_gpu_trace.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[0]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel1_datas.append(data_mean)
        elif idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(profile_path, "exec", f"{idx}_0_{kidx}_cuda_gpu_trace.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[1]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel2_datas.append(data_mean)
        elif idx == 2:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_parallel_statistics(data, [kname1, kname2])
                    parallel_datas.append(data_mean)
        elif idx == 3:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]
                    
                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_hfuse")
                    hfuse_datas.append(data_mean)
        elif idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(profile_path, "exec", f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_bfuse")
                    bfuse_datas.append(data_mean)

    return kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas
#-----------------------------------------------------------------------------------------------
def print_metric_cases(name, analysis):
    print(f"================================================== {name.upper()} CASES ==================================================")

    cases       = []
    issues      = []
    stalls      = []
    throughputs = []
    bfuse_perf  = []

    for bi, ci, bfuse_exec in analysis:
        # case A x B
        cases.append([bi, ci])

        # Issue rate utilization
        issue = [metric_datas[bi]["issue"], metric_datas[ci]["issue"]]
        issues.append(issue)

        # Stall reasons
        stall = [[metric_datas[bi]["stall"][0], metric_datas[bi]["stall"][1]],
                 [metric_datas[ci]["stall"][0], metric_datas[ci]["stall"][1]]]
        stalls.append(stall)

        # Throughput
        throughput = [[metric_datas[bi]["throughput"][0], metric_datas[bi]["throughput"][1]],
                      [metric_datas[ci]["throughput"][0], metric_datas[ci]["throughput"][1]]]
        throughputs.append(throughput)

        # BFuse performance
        bfuse_perf.append(bfuse_exec)

    # Print Datas
    print("<Cases>")
    print(f"total num: {len(cases)}")
    print("")
    for idx, (c, i, s, t) in enumerate(zip(cases, issues, stalls, throughputs)):
        if idx >= 5:
            print("Too many cases...")
            break

        issue_0   = f"{i[0]:.2f}"
        issue_1   = f"{i[1]:.2f}"
        stall_0   = f"{s[0][0]:.2f}"
        stall_1   = f"{s[1][0]:.2f}"
        compute_0 = f"{t[0][0]:.2f}"
        compute_1 = f"{t[1][0]:.2f}"
        memory_0  = f"{t[0][1]:.2f}"
        memory_1  = f"{t[1][1]:.2f}"

        issue_diff = f"{i[0] - i[1]:.2f}"
        if s[0][1] == s[1][1]:
            stall_diff   = f"{i[0] - i[1]:.2f}"
            stall_reason = s[0][1]
        else:
            stall_diff   = "NaN"
            stall_reason = "DIFF"
        compute_diff = f"{t[0][0] - t[1][0]:.2f}"
        memory_diff  = f"{t[0][1] - t[1][1]:.2f}"

        print(f"case (conv2d_{c[0]} x conv2d_{c[1]}): ")
        print(f" - conv2d_{c[0]:<2d}'s issue: {issue_0:>6s}, stall: {stall_0:>6s}({s[0][1]:>4s}), throughput: {compute_0:>6s}/{memory_0:>6s}")
        print(f" - conv2d_{c[1]:<2d}'s issue: {issue_1:>6s}, stall: {stall_1:>6s}({s[1][1]:>4s}), throughput: {compute_1:>6s}/{memory_1:>6s}")
        print(f" - differences issue: {issue_diff:>6s}, stall: {stall_diff:>6s}({stall_reason:>4s}), throughput: {compute_diff:>6s}/{memory_diff:>6s}")
        print(f" - bfuse perf: {bfuse_perf[idx]:.2f}")
#-----------------------------------------------------------------------------------------------
def draw_png(name, kernel1, kernel2, materials, parallel, hfuse, bfuse):
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
        axes[bi].set_xticks(index, axes_m, fontsize=3)

        # Set limitation of y axis
        axes[bi].set_ylim(0, 2)

        # Set major & minor yticks
        axes[bi].yaxis.set_major_locator(ticker.MultipleLocator(1))
        axes[bi].yaxis.set_major_formatter("{x:.0f}")

        # Set major & minor yticklabels
        yticks = axes[bi].get_yticks()
        axes[bi].set_yticklabels(yticks, fontdict={"fontsize": 3})

        axes[bi].set_ylabel('Speed up', fontsize=3)
        axes[bi].legend(ncol=1, loc="upper left", fontsize=2)

    # Save figure
    f.tight_layout()

    exec_figure = os.path.join(output_path, f"{name}.png")
    plt.savefig(exec_figure, dpi=500)
#-----------------------------------------------------------------------------------------------
def analysis_correlation(analysis):
    issue_diffs = []
    bfuse_execs = []
    for bi, ci, bfuse_exec in analysis:
        issue_diff = abs(metric_datas[bi]["issue"] - metric_datas[ci]["issue"])
        issue_diffs.append(issue_diff)
        bfuse_execs.append(bfuse_exec)

    cor, p_value = stats.pearsonr(issue_diffs, bfuse_execs)

    print("")
    print("<Correlation Coefficient>")
    print(f"Correlation: {cor:.4f}")
    print(f"P-value    : {p_value:.4f}")
#-----------------------------------------------------------------------------------------------
def draw_cases(func, name, kernel1, kernel2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas):
    materials = []
    parallel  = []
    hfuse     = []
    bfuse     = []
    analysis  = []

    for bi in range(len(kernel1)):
        for ci in range(len(kernel2)):
            serial_exec   = kernel1_datas[bi] + kernel2_datas[ci]
            parallel_exec = serial_exec / parallel_datas[bi * len(kernel2) + ci]
            hfuse_exec    = serial_exec / hfuse_datas[bi * len(kernel2) + ci]
            bfuse_exec    = serial_exec / bfuse_datas[bi * len(kernel2) + ci]

            materials.append("%s x %s" % (kernel1[bi], kernel2[ci]))
            if func(bfuse_exec):
                parallel.append(parallel_exec)
                hfuse.append(hfuse_exec)
                bfuse.append(bfuse_exec)
                analysis.append([bi, ci, bfuse_exec])
            else:
                parallel.append(0)
                hfuse.append(0)
                bfuse.append(0)

    # Print metric cases
    print_metric_cases(name, analysis)

    # Analysis metrics' correalation
    analysis_correlation(analysis)

    # Draw figure
    # draw_png(name, kernel1, kernel2, materials, parallel, hfuse, bfuse)
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

    # Draw low cases
    draw_cases(lambda x: x < 0.8, "low", kernel1, kernel2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

    # # Draw middle cases
    draw_cases(lambda x: x >= 0.8 and x < 1.2, "middle", kernel1, kernel2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

    # # Draw hight cases
    draw_cases(lambda x: x >= 1.4, "high", kernel1, kernel2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)
#-----------------------------------------------------------------------------------------------
def draw_metrics_graph(infoYAML, profile_path, output_path):
    pass
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)
    
    file_path   = "../../inception_v3"
    output_path = "."
    
    if not os.path.exists(file_path):
        logging.error("Given config path \"%s\" doesn't exist." % file_path)
        exit(1)

    # Parse YAML files
    config_path  = os.path.join(file_path, "config")
    profile_path = "."
    info_path    = os.path.join(config_path, "info.yaml")

    with open(info_path) as f:
        yaml_info = yaml.safe_load(f)

    draw_exec_graph(yaml_info, profile_path, output_path)
    draw_metrics_graph(yaml_info, profile_path, output_path)
#-----------------------------------------------------------------------------------------------