#!/usr/bin/python3

import os
import yaml, csv
import logging
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
import ncu_report
#-----------------------------------------------------------------------------------------------
# Settings
warmup_trials  = 0
metrics_trials = 1
exec_trials    = 30
#-----------------------------------------------------------------------------------------------
def check_config_valid(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    return len(fusion_sets) == 2
#-----------------------------------------------------------------------------------------------
def print_metrics(metrics_path):
    # Select arbitarily report file
    report_path = os.path.join(metrics_path, f"0_0_0.ncu-rep")
    ncu_context = ncu_report.load_report(report_path)

    ncu_range  = ncu_context.range_by_idx(0)
    ncu_action = ncu_range.action_by_idx(0)

    for name in ncu_action.metric_names():
        metric = ncu_action[name]

        # if metric.metric_type() == ncu_report.IMetric.MetricType_COUNTER:
        # if metric.metric_type() == ncu_report.IMetric.MetricType_RATIO:
        # if metric.metric_type() == ncu_report.IMetric.MetricType_THROUGHPUT:
        print(f"{metric.name():130s}: {metric.description()} ({metric.unit()})")
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
        logging.error("Number of kernel_names must be 2.")
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
def preprocess_exec(infoYAML, exec_path):
    # Parse YAML
    fusion_sets  = infoYAML["FusionSet"]
    kernel1_size = len(fusion_sets[0]["Set"])
    kernel2_size = len(fusion_sets[1]["Set"])

    test_methology = ["kernel1", "kernel2", "parallel", "hfuse", "bfuse"]

    kernel1_datas  = []
    kernel2_datas  = []
    parallel_datas = []
    hfuse_datas    = []
    bfuse_datas    = []

    for idx in range(len(test_methology)):
        if idx == 0:
            for kidx in range(kernel1_size):
                file_path = os.path.join(exec_path, f"{idx}_{kidx}_0_cuda_gpu_trace.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[0]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel1_datas.append(data_mean)
        elif idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join(exec_path, f"{idx}_0_{kidx}_cuda_gpu_trace.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[1]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel2_datas.append(data_mean)
        elif idx == 2:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(exec_path, f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_parallel_statistics(data, [kname1, kname2])
                    parallel_datas.append(data_mean)
        elif idx == 3:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(exec_path, f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]
                    
                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_hfuse")
                    hfuse_datas.append(data_mean)
        elif idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join(exec_path, f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_bfuse")
                    bfuse_datas.append(data_mean)

    return kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas
#-----------------------------------------------------------------------------------------------
def preprocess_metrics(infoYAML, metrics_path):
    # Parse YAML
    fusion_sets  = infoYAML["FusionSet"]
    kernel1_size = len(fusion_sets[0]["Set"])
    kernel2_size = len(fusion_sets[1]["Set"])

    kernel1_metrics  = []
    kernel2_metrics  = []
    parallel_metrics = []
    hfuse_metrics    = []
    bfuse_metrics    = []
    # metrics_list     = [# Stall reason
    #                     "smsp__average_warp_latency_per_inst_issued.ratio",
    #                     "smsp__average_warps_active_per_inst_executed.ratio",
    #                     "smsp__inst_issued.sum",
    #                     "smsp__inst_executed.sum",
    #                     # Issue rate
    #                     "sm__inst_issued.avg.per_cycle_active",
    #                     "sm__inst_executed.avg.per_cycle_active",
    #                     "sm__cycles_active.avg",
    #                    ]
    metrics_list = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",                 # Compute (SM) Throughput [%]
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", # Memory Throughput       [%]
        "l1tex__throughput.avg.pct_of_peak_sustained_active",               # L1/TEX Cache Throughput [%]
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",                # L2 Cache Throughput     [%]
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",           # DRAM Throughput         [%]
        "gpu__time_duration.sum",                                           # Duration                [msecond]
        "gpc__cycles_elapsed.max",                                          # Elapsed Cycles          [cycle]
        "sm__cycles_active.avg",                                            # SM Active Cycles        [cycle]
        "gpc__cycles_elapsed.avg.per_second",                               # SM Frequency            [cycle/nsecond]
        "dram__cycles_elapsed.avg.per_second",                              # DRAM Frequency          [cycle/nsecond]
    ]

    # Kernel 1
    for i1 in range(kernel1_size):
        report_path = os.path.join(metrics_path, f"0_{i1}_0.ncu-rep")
        ncu_context = ncu_report.load_report(report_path)
        ncu_range   = ncu_context.range_by_idx(0)
        ncu_action  = ncu_range.action_by_idx(0)

        metrics = []
        for m in metrics_list:
            metrics.append(ncu_action[m])
        kernel1_metrics.append(metrics)

    # Kernel 2
    for i2 in range(kernel2_size):
        report_path = os.path.join(metrics_path, f"1_0_{i2}.ncu-rep")
        ncu_context = ncu_report.load_report(report_path)
        ncu_range   = ncu_context.range_by_idx(0)
        ncu_action  = ncu_range.action_by_idx(0)

        metrics = []
        for m in metrics_list:
            metrics.append(ncu_action[m])
        kernel2_metrics.append(metrics)

    # HFuse
    # for i1 in range(kernel1_size):
    #     temp_list = []
    #     for i2 in range(kernel2_size):
    #         # TODO:
    #         if i1 > i2:
    #             i1, i2 = i2, i1
    #         report_path = os.path.join(metrics_path, f"3_{i1}_{i2}.ncu-rep")
    #         ncu_context = ncu_report.load_report(report_path)
    #         ncu_range   = ncu_context.range_by_idx(0)
    #         ncu_action  = ncu_range.action_by_idx(0)

    #         metrics = []
    #         for m in metrics_list:
    #             metrics.append(ncu_action[m])
    #         temp_list.append(metrics)
    #     hfuse_metrics.append(temp_list)

    # BFuse
    for i1 in range(kernel1_size):
        temp_list = []
        for i2 in range(kernel2_size):
            report_path = os.path.join(metrics_path, f"4_{i1}_{i2}.ncu-rep")
            ncu_context = ncu_report.load_report(report_path)
            ncu_range   = ncu_context.range_by_idx(0)
            ncu_action  = ncu_range.action_by_idx(0)

            metrics = []
            for m in metrics_list:
                metrics.append(ncu_action[m])
            temp_list.append(metrics)
        bfuse_metrics.append(temp_list)

    return metrics_list, kernel1_metrics, kernel2_metrics, parallel_metrics, hfuse_metrics, bfuse_metrics
#-----------------------------------------------------------------------------------------------
def collect_datas_with_condition(infoYAML, exec_path, metrics_path, condition):
    # Preprocess datas
    kernel1_exec, kernel2_exec, parallel_exec, hfuse_exec, bfuse_exec                         = preprocess_exec(infoYAML, exec_path)
    metrics, kernel1_metrics, kernel2_metrics, parallel_metrics, hfuse_metrics, bfuse_metrics = preprocess_metrics(infoYAML, metrics_path)

    # Parse YAML
    fusion_sets  = infoYAML["FusionSet"]
    kernel1_size = len(fusion_sets[0]["Set"])
    kernel2_size = len(fusion_sets[1]["Set"])

    cases    = []
    kernel1  = []
    kernel2  = []
    parallel = []
    hfuse    = []
    bfuse    = []

    for i1 in range(kernel1_size):
        for i2 in range(kernel2_size):
            serial_data   = kernel1_exec[i1] + kernel2_exec[i2]
            parallel_data = serial_data / parallel_exec[i1 * kernel2_size + i2]
            hfuse_data    = serial_data / hfuse_exec[i1 * kernel2_size + i2]
            bfuse_data    = serial_data / bfuse_exec[i1 * kernel2_size + i2]

            if condition(bfuse_data):
                cases.append([i1, i2])
                kernel1.append({"exec": kernel1_exec[i1], "metrics": kernel1_metrics[i1]})
                kernel2.append({"exec": kernel2_exec[i2], "metrics": kernel2_metrics[i2]})
                # parallel.append({"exec": parallel_exec[i1 * kernel2_size + i2], "metrics": parallel_metrics[i1 * kernel2_size + i2]})
                # hfuse.append({"exec": hfuse_exec[i1 * kernel2_size + i2], "metrics": hfuse_metrics[i1 * kernel2_size + i2]})
                bfuse.append({"exec": bfuse_exec[i1 * kernel2_size + i2], "metrics": bfuse_metrics[i1][i2]})

    return cases, kernel1, kernel2, parallel, hfuse, bfuse
#-----------------------------------------------------------------------------------------------
def draw_scatter(cases, kernel1, kernel2, parallel, hfuse, bfuse, output_path):
    # metrics_name = ["throttle_stall",
    #                 "depend_stall",
    #                 "issued_rate",
    #                 "execute_rate",
    #                 ]
    # metrics_idx  = [0, 1, 4, 5]
    metrics_name = [
        "Compute (SM) Throughput [%]",
        "Memory Throughput       [%]",
        "L1/TEX Cache Throughput [%]",
        "L2 Cache Throughput     [%]",
        "DRAM Throughput         [%]",
        "Duration                [msecond]",
        "Elapsed Cycles          [cycle]",
        "SM Active Cycles        [cycle]",
        "SM Frequency            [cycle/nsecond]",
        "DRAM Frequency          [cycle/nsecond]",
    ]
    metrics_idx = [i for i in range(len(metrics_name))]

    ###############
    # figure 1
    ###############

    # Align collected datas
    perf_list = []
    k1_list   = [[] for _ in range(len(metrics_name))]
    k2_list   = [[] for _ in range(len(metrics_name))]
    diff_list = [[] for _ in range(len(metrics_name))]
    for case, k1_data, k2_data, bfuse_data in zip(cases, kernel1, kernel2, bfuse):
        serial_exec = k1_data["exec"] + k2_data["exec"]
        perf        = serial_exec / bfuse_data["exec"]
        perf_list.append(perf)

        for idx in range(len(metrics_name)):
            m_idx       = metrics_idx[idx]
            k1_value    = k1_data["metrics"][m_idx].value()
            k2_value    = k2_data["metrics"][m_idx].value()
            bfuse_value = bfuse_data["metrics"][m_idx].value()

            if k1_value + k2_value == 0:
                bfuse_diff = 0
            # elif idx == 0 or idx == 1:
            #     s_idx      = m_idx + 2
            #     k1_inst    = k1_data["metrics"][s_idx].value()
            #     k2_inst    = k2_data["metrics"][s_idx].value()
            #     bfuse_inst = bfuse_data["metrics"][s_idx].value()
            #     bfuse_diff = bfuse_value / ((k1_value * k1_inst + k2_value * k2_inst) / (k1_inst + k2_inst))
            # elif idx == 2 or idx == 3:
            #     s_idx      = 6
            #     k1_inst    = k1_data["metrics"][s_idx].value()
            #     k2_inst    = k2_data["metrics"][s_idx].value()
            #     bfuse_inst = bfuse_data["metrics"][s_idx].value()
            #     bfuse_diff = bfuse_value / ((k1_value * k1_inst + k2_value * k2_inst) / (k1_inst + k2_inst))
            elif idx in [5, 6, 7]:
                bfuse_diff  = bfuse_value / (k1_value + k2_value)
            else:
                bfuse_diff  = bfuse_value / ((k1_value + k2_value) / 2)

            k1_list[idx].append(k1_value)
            k2_list[idx].append(k2_value)
            diff_list[idx].append(bfuse_diff)
    
    # Figure settings
    font = {"fontsize": 5, "fontweight": "bold"}
    col = 3
    row = math.ceil(len(metrics_name) / col) + 1

    # Draw figure (metrics)
    for idx, m_name in enumerate(metrics_name):
        plt.subplot(row, col, idx + 1)
        plt.title(f"{m_name}", fontdict=font)
        if idx in [5, 6, 7]:
            plt.scatter(perf_list, diff_list[idx], c="#FF7F00", s=0.5**2)
        else:
            plt.scatter(perf_list, diff_list[idx], s=0.5**2)
        plt.axhline(1, color="red", linestyle="--", linewidth=0.5)
        plt.tick_params(axis="x", labelsize=5)
        plt.tick_params(axis="y", labelsize=5)
    
    # Draw figure (data distribution)
    plt.subplot(row, col, len(metrics_name) + 1)
    plt.title("Distribution", fontdict=font)
    plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
    plt.tick_params(axis="x", labelsize=5)
    plt.tick_params(axis="y", labelsize=5)

    # Save figure
    file_path = os.path.join(output_path, "figure_1.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close()

    # Print statistics
    print(f"==== STATISTICS ====")
    print("---------------------------------------")
    print(f"Total num: {len(cases)}")
    print(f"Perf max:  {max(perf_list):.2f}x")
    print(f"Perf min:  {min(perf_list):.2f}x")
    print(f"Perf avr:  {np.mean(perf_list):.2f}x")
    print(f"Perf std:  {np.std(perf_list):.2f}")
    print("---------------------------------------")
    for idx, m_name in enumerate(metrics_name):
        print(f"{m_name:40s}: {np.mean(diff_list[idx]):.2f}x")
    print("---------------------------------------")
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", action="store_true", default=False, dest="show",
                        help="options for printing collected metrics")
    parser.add_argument("file", action="store", help="path of generated test-suite")
    parser.add_argument("-o", action="store", default=".", dest="output",
                        help="output path of figure")

    # Get arguments
    args        = parser.parse_args()
    show        = args.show
    file_path   = args.file
    output_path = args.output

    # Check the neccessary directories/files exist
    if not os.path.exists(file_path):
        logging.error("Given path \"%s\" doesn't exist." % file_path)
        exit(1)

    profile_path = os.path.join(file_path, "profile")
    exec_path    = os.path.join(file_path, "profile", "exec")
    metrics_path = os.path.join(file_path, "profile", "metrics")
    if not os.path.exists(profile_path):
        logging.error("Given path \"%s\" doesn't exist." % profile_path)
        exit(1)
    if not os.path.exists(exec_path):
        logging.error("Given path \"%s\" doesn't exist." % exec_path)
        exit(1)
    if not os.path.exists(metrics_path):
        logging.error("Given path \"%s\" doesn't exist." % metrics_path)
        exit(1)

    # Parse YAML files
    config_path = os.path.join(file_path, "config")
    info_path   = os.path.join(config_path, "info.yaml")

    with open(info_path) as f:
        yaml_info = yaml.safe_load(f)

    if not check_config_valid(yaml_info):
        logging.error("Given config file \"%s\" is invalid." % file_path)
        exit(1)

    # Print metrics list
    if show:
        print_metrics(metrics_path)
        exit(0)

    # Collect datas with condition
    cases, kernel1, kernel2, parallel, hfuse, bfuse = collect_datas_with_condition(yaml_info, exec_path, metrics_path, lambda x: True)
    
    # draw figure
    draw_scatter(cases, kernel1, kernel2, parallel, hfuse, bfuse, output_path)