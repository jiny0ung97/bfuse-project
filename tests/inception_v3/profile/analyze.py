#!/usr/bin/python3

import os
import logging
import yaml, csv

import numpy as np
import ncu_report
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
def preprocess_datas(infoYAML):
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
                file_path = os.path.join("exec", f"{idx}_{kidx}_0_cuda_gpu_trace.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[0]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel1_datas.append(data_mean)
        elif idx == 1:
            for kidx in range(kernel2_size):
                file_path = os.path.join("exec", f"{idx}_0_{kidx}_cuda_gpu_trace.csv")
                data, _   = parse_csv(file_path)
                kname     = fusion_sets[1]["Set"][kidx]

                _, _, data_mean, _ = get_single_statistics(data, kname)
                kernel2_datas.append(data_mean)
        elif idx == 2:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join("exec", f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_parallel_statistics(data, [kname1, kname2])
                    parallel_datas.append(data_mean)
        elif idx == 3:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join("exec", f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]
                    
                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_hfuse")
                    hfuse_datas.append(data_mean)
        elif idx == 4:
            for kidx1 in range(kernel1_size):
                for kidx2 in range(kernel2_size):
                    file_path = os.path.join("exec", f"{idx}_{kidx1}_{kidx2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][kidx1]
                    kname2    = fusion_sets[1]["Set"][kidx2]

                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_bfuse")
                    bfuse_datas.append(data_mean)

    return kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas
#-----------------------------------------------------------------------------------------------
def preprocess_metrics(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(1)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])

    kernel1_metrics = []
    kernel2_metrics = []
    metrics_list    = ["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
                       ]

    for i0 in range(kernel1_size):
        report_path = os.path.join("metrics", f"0_{i0}_0.ncu-rep")
        ncu_context = ncu_report.load_report(report_path)

        ncu_range  = ncu_context.range_by_idx(0)
        ncu_action = ncu_range.action_by_idx(0)

        metrics = []
        for m in metrics_list:
            metrics.append(ncu_action[m])
        kernel1_metrics.append(metrics)

    for i1 in range(kernel2_size):
        report_path = os.path.join("metrics", f"1_0_{i1}.ncu-rep")
        ncu_context = ncu_report.load_report(report_path)

        ncu_range  = ncu_context.range_by_idx(0)
        ncu_action = ncu_range.action_by_idx(0)

        metrics = []
        for m in metrics_list:
            metrics.append(ncu_action[m])
        kernel2_metrics.append(metrics)

    return metrics_list, kernel1_metrics, kernel2_metrics
#-----------------------------------------------------------------------------------------------
def preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(1)

    kernel1 = fusion_sets[0]["Set"]
    kernel2 = fusion_sets[1]["Set"]

    # materials = []
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

            # materials.append("%s x %s" % (kernel1[bi], kernel2[ci]))
            if cond(bfuse_exec):
                parallel.append(parallel_exec)
                hfuse.append(hfuse_exec)
                bfuse.append(bfuse_exec)
                analysis.append([bi, ci, bfuse_exec])
            else:
                parallel.append(0)
                hfuse.append(0)
                bfuse.append(0)

    return parallel, hfuse, bfuse, analysis
#-----------------------------------------------------------------------------------------------
def analyze_cond(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics):
    paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)
    for bi, ci, bfuse_exec in analysis:
        metric_diffs = []
        for idx in range(len(kernel1_metrics[bi])):
            diff = f"{abs(kernel1_metrics[bi][idx].value() - kernel2_metrics[ci][idx].value()):.2f}"
            metric_diffs.append(f" - {kernel1_metrics[bi][idx].name():100s} diff: {diff:5s} ({kernel1_metrics[bi][idx].unit()})")

    print(f"================================================== {name.upper()} CASES ==================================================")
    for idx, [bi, ci, bfuse_exec] in enumerate(analysis):
        if idx >= 10:
            print("Too many cases...")
            break
        
        print(f"case (conv2d_{bi} x conv2d_{ci}): ")
        for diff in metric_diffs:
            print(diff)
#-----------------------------------------------------------------------------------------------
def analyze_report(infoYAML):
    # Preprocess datas
    kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas = preprocess_datas(infoYAML)
    metrics_list, kernel1_metrics, kernel2_metrics = preprocess_metrics(infoYAML)

    # Case low
    analyze_cond(infoYAML, "low", lambda x: x < 0.8, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics)

    # Case middle
    analyze_cond(infoYAML, "middle", lambda x: x >= 0.8 and x < 1.2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics)

    # Case high
    analyze_cond(infoYAML, "high", lambda x: x >= 1.2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics)
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)
    
    # Settings
    exec_path    = "./exec"
    metrics_path = "./metrics"
    config_path  = "../config"
    info_file    = os.path.join(config_path, "info.yaml")

    if not os.path.exists(exec_path):
        logging.error("Given config path \"%s\" doesn't exist." % exec_path)
        exit(1)
    if not os.path.exists(metrics_path):
        logging.error("Given config path \"%s\" doesn't exist." % metrics_path)
        exit(1)

    # Parse YAML files
    with open(info_file) as f:
        yaml_info = yaml.safe_load(f)

    # Analyze ncu report results
    analyze_report(yaml_info)
#-----------------------------------------------------------------------------------------------