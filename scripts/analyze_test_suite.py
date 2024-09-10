#!/usr/bin/python3

import os
import yaml, csv
import logging
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
import ncu_report
#-----------------------------------------------------------------------------------------------
# Settings
warmup_trials  = 3
metrics_trials = 1
exec_trials    = 30

# Conditions
lsu_rate_cond     = 4.5
lsu_sum_cond_up   = 33
lsu_sum_cond_down = 0
fma_rate_cond     = 4.5
fma_sum_cond_up   = 21
fma_sum_cond_down = 0
# lsu_rate_cond     = 4.5
# lsu_sum_cond_up   = 100
# lsu_sum_cond_down = 50
# fma_rate_cond     = 4.5
# fma_sum_cond_up   = 100
# fma_sum_cond_down = 14
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
                temp_list = []
                for kidx2 in range(kernel2_size):
                    if kidx2 < kidx1:
                        i1, i2 = kidx2, kidx1
                    else:
                        i1, i2 = kidx1, kidx2
                        
                    file_path = os.path.join(exec_path, f"{idx}_{i1}_{i2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][i1]
                    kname2    = fusion_sets[1]["Set"][i2]

                    _, _, data_mean, _ = get_parallel_statistics(data, [kname1, kname2])
                    temp_list.append(data_mean)
                parallel_datas.append(temp_list)
        elif idx == 3:
            for kidx1 in range(kernel1_size):
                temp_list = []
                for kidx2 in range(kernel2_size):
                    if kidx2 < kidx1:
                        i1, i2 = kidx2, kidx1
                    else:
                        i1, i2 = kidx1, kidx2

                    file_path = os.path.join(exec_path, f"{idx}_{i1}_{i2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][i1]
                    kname2    = fusion_sets[1]["Set"][i2]
                    
                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_hfuse")
                    temp_list.append(data_mean)
                hfuse_datas.append(temp_list)
        elif idx == 4:
            for kidx1 in range(kernel1_size):
                temp_list = []
                for kidx2 in range(kernel2_size):
                    if kidx2 < kidx1:
                        i1, i2 = kidx2, kidx1
                    else:
                        i1, i2 = kidx1, kidx2

                    file_path = os.path.join(exec_path, f"{idx}_{i1}_{i2}_cuda_gpu_trace.csv")
                    data, _   = parse_csv(file_path)
                    kname1    = fusion_sets[0]["Set"][i1]
                    kname2    = fusion_sets[1]["Set"][i2]

                    _, _, data_mean, _ = get_single_statistics(data, f"{kname1}_{kname2}_fused_bfuse")
                    temp_list.append(data_mean)
                bfuse_datas.append(temp_list)

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
    metrics_list = [
        #### GPU Spped Of Light Throughput #### 0
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",                            # Compute (SM) Throughput [%]
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",            # Memory Throughput       [%]
        "l1tex__throughput.avg.pct_of_peak_sustained_active",                          # L1/TEX Cache Throughput [%]
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",                           # L2 Cache Throughput     [%]
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",                      # DRAM Throughput         [%]
        "gpu__time_duration.sum",                                                      # Duration                [msecond]
        "gpc__cycles_elapsed.max",                                                     # Elapsed Cycles          [cycle]
        "sm__cycles_active.avg",                                                       # SM Active Cycles        [cycle]
        "gpc__cycles_elapsed.avg.per_second",                                          # SM Frequency            [cycle/nsecond]
        "dram__cycles_elapsed.avg.per_second",                                         # DRAM Frequency          [cycle/nsecond]
        #### Compute Workload Analysis #### 10
        "sm__inst_executed.avg.per_cycle_elapsed",                                     # Executed Ipc Elapsed [inst/cycle]
        "sm__inst_executed.avg.per_cycle_active",                                      # Executed Ipc Active  [inst/cycle]
        "sm__inst_issued.avg.per_cycle_active",                                        # Issued Ipc Active    [inst/cycle]
        "sm__instruction_throughput.avg.pct_of_peak_sustained_active",                 # SM Busy              [%]
        "sm__inst_issued.avg.pct_of_peak_sustained_active",                            # Issue Slots Busy     [%]
        #### Memory Workload Analysis #### 15
        "dram__bytes.sum.per_second",                                                  # Memory Throughput           [Gbyte/second]
        "l1tex__t_sector_hit_rate.pct",                                                # L1/TEX Hit Rate             [%]
        "lts__t_sector_hit_rate.pct",                                                  # L2 Hit Rate                 [%]
        "lts__average_gcomp_input_sector_success_rate.pct",                            # L2 Compression Success Rate [%]
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",     # Mem Busy                    [%]
        "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",    # Max Bandwidth               [%]
        "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",                     # Mem Pipes Busy              [%]
        "lts__average_gcomp_output_sector_compression_achieved_rate.ratio",            # L2 Compression Ratio
        #### Scheduler Statistics #### 23
        "smsp__warps_active.avg.per_cycle_active",                                     # Active Warps Per Scheduler   [warp]
        "smsp__warps_eligible.avg.per_cycle_active",                                   # Eligible Warps Per Scheduler [warp]
        "smsp__issue_active.avg.per_cycle_active",                                     # Issued Warp Per Scheduler
        "smsp__issue_inst0.avg.pct_of_peak_sustained_active",                          # No Eligible                  [%]
        "smsp__issue_active.avg.pct_of_peak_sustained_active",                         # One or More Eligible         [%]
        #### Warp State Statistics #### 28
        "smsp__average_warp_latency_per_inst_issued.ratio",                            # Warp Cycles Per Issued Instruction       [cycle]
        "smsp__average_warps_active_per_inst_executed.ratio",                          # Warp Cycles Per Executed Instruction     [cycle]
        "smsp__thread_inst_executed_per_inst_executed.ratio",                          # Avg. Active Threads Per Warp
        "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",                  # Avg. Not Predicated Off Threads Per Warp
        #### Instruction Statistics #### 32
        "smsp__inst_executed.sum",                                                     # Executed Instructions                    [inst]
        "smsp__inst_issued.sum",                                                       # Issued Instructions                      [inst]
        "smsp__inst_executed.avg",                                                     # Avg. Executed Instructions Per Scheduler [inst]
        "smsp__inst_issued.avg",                                                       # Avg. Issued Instructions Per Scheduler   [inst]
        #### Breakdown: Execution Instruction Mix #### 36
        "sass__inst_executed_per_opcode",
        #### Breakdown: Warp State (All Cycles) #### 37
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",            # Stall Barrier
        "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",   # Stall Branch Resolving
        "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",     # Stall Dispatch Stall
        "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",              # Stall Drain
        "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",           # Stall IMC Miss
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",        # Stall LG Throttle
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",    # Stall Long Scoreboard
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", # Stall Math Pipe Throttle
        "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",             # Stall Membar
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",       # Stall MIO Throttle
        "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",               # Stall Misc
        "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",     # Stall No Instruction
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",       # Stall Not Selected
        "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",           # Selected
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",   # Stall Short Scoreboard
        "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",           # Stall Sleeping
        "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",       # Stall Tex Throttle
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",               # Stall Wait
        #### Breakdown: Pipe Utilization (% of peak instructions executed) #### 55
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",                 # LSU
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",                 # FMA
        "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",                 # ALU
        "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",                 # ADU
        "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active",             # Uniform
        "sm__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_active",                 # CBU
        "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active",       # FMA (FP16)
        "sm__inst_executed_pipe_fp64_op_dmma.avg.pct_of_peak_sustained_active",        # FP64 (DMMA)
        "sm__inst_executed_pipe_fp64_op_fp64.avg.pct_of_peak_sustained_active",        # FP64 (FP64)
        "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",                 # TEX
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",      # Tensor (FP)
        "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",      # Tensor (INT
        "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",                  # XU
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
    for idx1 in range(kernel1_size):
        temp_list = []
        for idx2 in range(kernel2_size):
            if idx1 > idx2:
                i1, i2 = idx2, idx1
            else:
                i1, i2 = idx1, idx2

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
def collect_datas_with_condition(infoYAML, exec_path, metrics_path, condition, inst_cond, pipe_cond):
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
        for i2 in range(i1, kernel2_size):
            if i1 > i2:
                continue

            serial_data   = kernel1_exec[i1] + kernel2_exec[i2]
            parallel_data = serial_data / parallel_exec[i1][i2]
            hfuse_data    = serial_data / hfuse_exec[i1][i2]
            bfuse_data    = serial_data / bfuse_exec[i1][i2]

            if not (bfuse_data - hfuse_data >= 0.4):
                continue

            if not condition(bfuse_data):
                continue

            if inst_cond:
                k1_executed_inst    = kernel1_metrics[i1][32].value()
                k1_issued_inst      = kernel1_metrics[i1][33].value()
                k2_executed_inst    = kernel2_metrics[i2][32].value()
                k2_issued_inst      = kernel2_metrics[i2][33].value()
                bfuse_executed_inst = bfuse_metrics[i1][i2][32].value()
                bfuse_issued_inst   = bfuse_metrics[i1][i2][33].value()

                diff_executed_inst = bfuse_executed_inst / (k1_executed_inst + k2_executed_inst)
                diff_issued_inst   = bfuse_issued_inst / (k1_issued_inst + k2_issued_inst)

                if not (0.9 <= diff_executed_inst < 1.1 and 0.9 <= diff_issued_inst < 1.1):
                    continue
                # if not valid and (0.9 <= diff_executed_inst < 1.1 and 0.9 <= diff_issued_inst < 1.1):
                #     continue

            if pipe_cond:
                k1_lsu    = kernel1_metrics[i1][55].value()
                k1_fma    = kernel1_metrics[i1][56].value()
                k1_weight = kernel1_metrics[i1][5].value()

                k2_lsu    = kernel2_metrics[i2][55].value()
                k2_fma    = kernel2_metrics[i2][56].value()
                k2_weight = kernel2_metrics[i2][5].value()

                k1_lsu_weighted = k1_lsu * (k1_weight / (k1_weight + k2_weight))
                k1_fma_weighted = k1_fma * (k1_weight / (k1_weight + k2_weight))
                k2_lsu_weighted = k2_lsu * (k2_weight / (k1_weight + k2_weight))
                k2_fma_weighted = k2_fma * (k2_weight / (k1_weight + k2_weight))

                if k1_lsu_weighted == 0 or k2_lsu_weighted == 0:
                    continue

                diff_lsu = k1_lsu_weighted / k2_lsu_weighted
                diff_fma = k1_fma_weighted / k2_fma_weighted
                sum_lsu  = k1_lsu_weighted + k2_lsu_weighted
                sum_fma  = k1_fma_weighted + k2_fma_weighted

                # if not (1/lsu_rate_cond <= diff_lsu < lsu_rate_cond and lsu_sum_cond_down <= sum_lsu < lsu_sum_cond_up) or \
                #     not (1/fma_rate_cond <= diff_fma < fma_rate_cond and fma_sum_cond_down <= sum_fma < fma_sum_cond_up):
                #     continue
                if not (lsu_sum_cond_down <= sum_lsu < lsu_sum_cond_up) or \
                    not (fma_sum_cond_down <= sum_fma < fma_sum_cond_up):
                    continue

            cases.append([i1, i2])
            kernel1.append({"exec": kernel1_exec[i1], "metrics": kernel1_metrics[i1]})
            kernel2.append({"exec": kernel2_exec[i2], "metrics": kernel2_metrics[i2]})
            parallel.append({"exec": parallel_exec[i1][i2], "metrics": None})
            hfuse.append({"exec": hfuse_exec[i1][i2], "metrics": None})
            bfuse.append({"exec": bfuse_exec[i1][i2], "metrics": bfuse_metrics[i1][i2]})

    return cases, kernel1, kernel2, parallel, hfuse, bfuse
#-----------------------------------------------------------------------------------------------
def draw_scatter(figure_name, output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list, ylim_list=None):
    # Align collected datas
    perf_list = []
    k1_list   = [[] for _ in range(len(metrics_name))]
    k2_list   = [[] for _ in range(len(metrics_name))]
    diff_list = [[] for _ in range(len(metrics_name))]
    for case, k1_data, k2_data, bfuse_data in data_pack:
        serial_exec = k1_data["exec"] + k2_data["exec"]
        perf        = serial_exec / bfuse_data["exec"]
        perf_list.append(perf)

        for idx in range(len(metrics_name)):
            m_idx       = metrics_idx[idx]
            k1_value    = k1_data["metrics"][m_idx].value()
            k2_value    = k2_data["metrics"][m_idx].value()
            bfuse_value = bfuse_data["metrics"][m_idx].value()

            if k1_value == 0 and k2_value == 0:
                bfuse_diff = float("nan")
            elif idx in second_list:
                k1_weight  = k1_data["metrics"][5].value()
                k2_weight  = k2_data["metrics"][5].value()
                bfuse_diff = bfuse_value / np.average([k1_value, k2_value], weights=[k1_weight, k2_weight])
            elif idx in elapsed_list:
                k1_weight  = k1_data["metrics"][6].value()
                k2_weight  = k2_data["metrics"][6].value()
                bfuse_diff = bfuse_value / np.average([k1_value, k2_value], weights=[k1_weight, k2_weight])
            elif idx in active_list:
                k1_weight  = k1_data["metrics"][7].value()
                k2_weight  = k2_data["metrics"][7].value()
                bfuse_diff = bfuse_value / np.average([k1_value, k2_value], weights=[k1_weight, k2_weight])
            elif idx in sum_list:
                bfuse_diff = bfuse_value / (k1_value + k2_value)
            else:
                bfuse_diff = bfuse_value / ((k1_value + k2_value) / 2)

            k1_list[idx].append(k1_value)
            k2_list[idx].append(k2_value)
            diff_list[idx].append(bfuse_diff)
    
    # Figure settings
    font = {"fontsize": 5, "fontweight": "bold"}
    col = 3
    row = math.ceil(len(metrics_name) / col) + 1

    # Draw figure (metrics)
    # plt.figure(figsize=(15, 15))
    for idx, m_name in enumerate(metrics_name):
        plt.subplot(row, col, idx + 1)
        plt.title(f"{m_name}", fontdict=font)
        if idx in low_is_best_list:
            color = "#FF7F00"
        else:
            color = "#1E90FF"
        plt.scatter(perf_list, diff_list[idx], c=color, s=0.5**2)
        plt.axhline(1, color="red", linestyle="--", linewidth=0.5)
        plt.tick_params(axis="x", labelsize=5)
        plt.tick_params(axis="y", labelsize=5)
        if ylim_list and ylim_list[idx]:
            plt.ylim(ylim_list[idx])
    
    # Draw figure (data distribution)
    plt.subplot(row, col, len(metrics_name) + 1)
    plt.title("Distribution", fontdict=font)
    plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
    plt.tick_params(axis="x", labelsize=5)
    plt.tick_params(axis="y", labelsize=5)

    # Save figure
    file_path = os.path.join(output_path, f"{figure_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close()

    # Print statistics
    print(f"==== {figure_name.upper()} ====")
    print("---------------------------------------")
    print(f"Total num: {len(perf_list)}")
    print(f"Perf max:  {max(perf_list):.2f}x")
    print(f"Perf min:  {min(perf_list):.2f}x")
    print(f"Perf avr:  {np.mean(perf_list):.2f}x")
    print(f"Perf std:  {np.std(perf_list):.2f}")
    print("---------------------------------------")
    for idx, m_name in enumerate(metrics_name):
        print(f"{m_name:50s}: {np.mean(diff_list[idx]):.2f}x")
    print("---------------------------------------")
#-----------------------------------------------------------------------------------------------
def draw_stall_breakdown_scatter(figure_name, output_path, metrics_name, metrics_idx, data_pack, low_is_best_list, ylim_list=None):
    # Align collected datas
    perf_list  = []
    k1_list    = [[] for _ in range(len(metrics_name))]
    k2_list    = [[] for _ in range(len(metrics_name))]
    bfuse_list = [[] for _ in range(len(metrics_name))]
    diff_list  = [[] for _ in range(len(metrics_name))]
    mean_list  = [[] for _ in range(len(metrics_name))]
    imp_list   = [[] for _ in range(len(metrics_name))]
    for case, k1_data, k2_data, bfuse_data in data_pack:
        serial_exec = k1_data["exec"] + k2_data["exec"]
        perf        = serial_exec / bfuse_data["exec"]
        perf_list.append(perf)

        for idx in range(len(metrics_name)):
            m_idx       = metrics_idx[idx]
            k1_value    = k1_data["metrics"][m_idx].value()
            k2_value    = k2_data["metrics"][m_idx].value()
            bfuse_value = bfuse_data["metrics"][m_idx].value()

            k1_weight = k1_data["metrics"][5].value()
            k2_weight = k2_data["metrics"][5].value()

            k1_value_ = k1_value * k1_weight / (k1_weight + k2_weight)
            k2_value_ = k2_value * k2_weight / (k1_weight + k2_weight)

            mean_value = np.average([k1_value, k2_value], weights=[k1_weight, k2_weight])
            diff_value = abs(k1_value_ - k2_value_)
            if bfuse_value == 0:
                imp_value = float("nan")
            else:
                imp_value = mean_value / bfuse_value

            k1_list[idx].append(k1_value)
            k2_list[idx].append(k2_value)
            # k1_list[idx].append(k1_value_)
            # k2_list[idx].append(k2_value_)
            bfuse_list[idx].append(bfuse_value)
            diff_list[idx].append(diff_value)
            mean_list[idx].append(mean_value)
            imp_list[idx].append(imp_value)
    
    # Figure settings
    font = {"fontsize": 5, "fontweight": "bold"}
    col = 3
    row = math.ceil(len(metrics_name) / col) + 1

    # Draw figure (metrics)
    plt.figure(figsize=(15, 15))
    for idx, m_name in enumerate(metrics_name):
        plt.subplot(row, col, idx + 1)
        plt.title(f"{m_name}", fontdict=font)
        # if idx in low_is_best_list:
        #     color = "#FF7F00"
        # else:
        #     color = "#1E90FF"
        # plt.scatter(perf_list, k1_list[idx], c="#FF7F00", s=0.5**2)
        # plt.scatter(perf_list, k2_list[idx], c="#1E90FF", s=0.5**2)
        plt.scatter(k1_list[idx], k2_list[idx], c=imp_list[idx], s=1**2, alpha=0.8)
        plt.tick_params(axis="x", labelsize=5)
        plt.tick_params(axis="y", labelsize=5)
        # if ylim_list and ylim_list[idx]:
        #     plt.ylim(ylim_list[idx])
        plt.colorbar(shrink=0.8, aspect=10)
    
    # Draw figure (data distribution)
    plt.subplot(row, col, len(metrics_name) + 1)
    plt.title("Distribution", fontdict=font)
    plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
    plt.tick_params(axis="x", labelsize=5)
    plt.tick_params(axis="y", labelsize=5)

    # Save figure
    file_path = os.path.join(output_path, f"{figure_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close()

    # Print statistics
    print(f"==== {figure_name.upper()} ====")
    print("---------------------------------------")
    print(f"Total num: {len(perf_list)}")
    print(f"Perf max:  {max(perf_list):.2f}x")
    print(f"Perf min:  {min(perf_list):.2f}x")
    print(f"Perf avr:  {np.mean(perf_list):.2f}x")
    print(f"Perf std:  {np.std(perf_list):.2f}")
    print("---------------------------------------")
    for idx, m_name in enumerate(metrics_name):
        print(f"{m_name:50s}: {np.mean(diff_list[idx]):.2f}x")
    print("---------------------------------------")
#-----------------------------------------------------------------------------------------------
def draw_compute_breakdown_scatter(figure_name, output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list, ylim_list=None):
    # Align collected datas
    perf_list   = []
    k1_list     = [[] for _ in range(len(metrics_name))]
    k2_list     = [[] for _ in range(len(metrics_name))]
    bfuse_list  = [[] for _ in range(len(metrics_name))]
    diff_list   = [[] for _ in range(len(metrics_name))]
    mean_list   = [[] for _ in range(len(metrics_name))]
    imp_list    = [[] for _ in range(len(metrics_name))]
    for case, k1_data, k2_data, bfuse_data in data_pack:
        serial_exec = k1_data["exec"] + k2_data["exec"]
        perf        = serial_exec / bfuse_data["exec"]
        perf_list.append(perf)

        for idx in range(len(metrics_name)):
            m_idx       = metrics_idx[idx]
            k1_value    = k1_data["metrics"][m_idx].value()
            k2_value    = k2_data["metrics"][m_idx].value()
            bfuse_value = bfuse_data["metrics"][m_idx].value()

            k1_weight = k1_data["metrics"][5].value()
            k2_weight = k2_data["metrics"][5].value()

            k1_value_ = k1_value * (k1_weight / (k1_weight + k2_weight))
            k2_value_ = k2_value * (k2_weight / (k1_weight + k2_weight))

            if k1_value_ == 0 and k2_value_ == 0:
                mean_value = float("nan")
            else:
                mean_value = np.average([k1_value, k2_value], weights=[k1_weight, k2_weight])
            diff_value = abs(k1_value_ - k2_value_)
            imp_value  = bfuse_value / mean_value

            # k1_list[idx].append(k1_value)
            # k2_list[idx].append(k2_value)
            k1_list[idx].append(k1_value_)
            k2_list[idx].append(k2_value_)
            bfuse_list[idx].append(bfuse_value)
            diff_list[idx].append(diff_value)
            mean_list[idx].append(mean_value)
            imp_list[idx].append(imp_value)
    
    # Figure settings
    font = {"fontsize": 5, "fontweight": "bold"}
    col = 3
    row = math.ceil(len(metrics_name) / col) + 1

    # Draw figure (metrics)
    plt.figure(figsize=(15, 15))
    for idx, m_name in enumerate(metrics_name):
        plt.subplot(row, col, idx + 1)
        plt.title(f"{m_name}", fontdict=font)
        if idx in low_is_best_list:
            color = "#FF7F00"
        else:
            color = "#1E90FF"
        plt.scatter(k1_list[idx], k2_list[idx], c=imp_list[idx], s=1**2, alpha=0.8)
        # plt.scatter(k1_list[idx], k2_list[idx], c=perf_list, s=1**2, alpha=0.8)
        plt.tick_params(axis="x", labelsize=5)
        plt.tick_params(axis="y", labelsize=5)
        # if ylim_list and ylim_list[idx]:
        #     plt.ylim(ylim_list[idx])
        plt.colorbar(shrink=0.8, aspect=10)
        # total_list = k1_list[idx] + k2_list[idx]
        if idx == 0:
            # plt.plot([min(total_list), max(total_list)], [min(total_list), max(total_list)], "--", color="red", linewidth=0.5) # y = x
            # plt.plot([min(total_list), max(total_list)], [min(total_list) * 1/lsu_rate_cond, max(total_list) * 1/lsu_rate_cond], "--", color="red", linewidth=0.5) # y = 1/rate_cond * x
        #     plt.plot([min(total_list) * 1/lsu_rate_cond, max(total_list) * 1/lsu_rate_cond], [min(total_list), max(total_list)], "--", color="red", linewidth=0.5) # y = rate_cond * x
            # plt.plot([lsu_sum_cond_down, 0], [0, lsu_sum_cond_down], "--", color="red", linewidth=0.5) # x + y = sum_cond_down
            if lsu_sum_cond_up != 100:
                plt.plot([lsu_sum_cond_up, 0], [0, lsu_sum_cond_up], "--", color="red", linewidth=0.5)     # x + y = sum_cond_up
            # plt.clim(0.6, 1.5)
        elif idx == 1:
        #     plt.plot([min(total_list), max(total_list)], [min(total_list), max(total_list)], "--", color="red", linewidth=0.5) # y = x
        #     plt.plot([min(total_list), max(total_list)], [min(total_list) * 1/fma_rate_cond, max(total_list) * 1/fma_rate_cond], "--", color="red", linewidth=0.5) # y = 1/rate_cond * x
        #     plt.plot([min(total_list) * 1/fma_rate_cond, max(total_list) * 1/fma_rate_cond], [min(total_list), max(total_list)], "--", color="red", linewidth=0.5) # y = rate_cond * x
            # plt.plot([fma_sum_cond_down, 0], [0, fma_sum_cond_down], "--", color="red", linewidth=0.5) # x + y = sum_cond_down
            if fma_sum_cond_up != 100:
                plt.plot([fma_sum_cond_up, 0], [0, fma_sum_cond_up], "--", color="red", linewidth=0.5)     # x + y = sum_cond_up
        #     plt.clim(0.6, 1.5)
        # if max(total_list) >= 80:
        #     plt.plot([100, 0], [0, 100], "--", color="red", linewidth=0.5) # x + y = 100
    
    # Draw figure (data distribution)
    plt.subplot(row, col, len(metrics_name) + 1)
    plt.title("Distribution", fontdict=font)
    plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
    plt.tick_params(axis="x", labelsize=5)
    plt.tick_params(axis="y", labelsize=5)

    # Save figure
    file_path = os.path.join(output_path, f"{figure_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close()

    # Print statistics
    print(f"==== {figure_name.upper()} ====")
    print("---------------------------------------")
    print(f"Total num: {len(perf_list)}")
    print(f"Perf max:  {max(perf_list):.2f}x")
    print(f"Perf min:  {min(perf_list):.2f}x")
    print(f"Perf avr:  {np.mean(perf_list):.2f}x")
    print(f"Perf std:  {np.std(perf_list):.2f}")
    print("---------------------------------------")
    for idx, m_name in enumerate(metrics_name):
        print(f"{m_name:50s}:")
        print(f" - K1:   {np.mean(k1_list[idx]):.2f}")
        print(f" - K2:   {np.mean(k1_list[idx]):.2f}")
        print(f" - Perf: {np.mean(imp_list[idx]):.2f}x")
    print("---------------------------------------")
#-----------------------------------------------------------------------------------------------
def draw_inst_breakdwon_hist(figure_name, output_path, data_pack, m_idx, xlim=None):
    # Initialize list
    metrics_name = []
    perf_list    = []
    k1_list      = []
    k2_list      = []
    bfuse_list   = []

    # Collect datas
    for case, k1_data, k2_data, bfuse_data in data_pack:
        serial_exec = k1_data["exec"] + k2_data["exec"]
        perf        = serial_exec / bfuse_data["exec"]
        perf_list.append(perf)

        # Kernel 1
        k1_dict            = {}
        k1_inst_per_opcode = k1_data["metrics"][m_idx]
        k1_num_opcodes     = k1_inst_per_opcode.num_instances()
        k1_opcodes         = k1_inst_per_opcode.correlation_ids()
        for i in range(k1_num_opcodes):
            op  = k1_opcodes.as_string(i)
            num = k1_inst_per_opcode.as_uint64(i)
            if op in k1_dict:
                k1_dict[op].append(num)
            else:
                k1_dict[op] = [num]
            
            # Append opcode name
            if op not in metrics_name:
                metrics_name.append(op)
        k1_list.append(k1_dict)

        # Kernel 2
        k2_dict            = {}
        k2_inst_per_opcode = k2_data["metrics"][m_idx]
        k2_num_opcodes     = k2_inst_per_opcode.num_instances()
        k2_opcodes         = k2_inst_per_opcode.correlation_ids()
        for i in range(k2_num_opcodes):
            op  = k2_opcodes.as_string(i)
            num = k2_inst_per_opcode.as_uint64(i)
            if op in k2_dict:
                k2_dict[op].append(num)
            else:
                k2_dict[op] = [num]

            # Append opcode name
            if op not in metrics_name:
                metrics_name.append(op)
        k2_list.append(k2_dict)

        # BFuse
        bfuse_dict            = {}
        bfuse_inst_per_opcode = bfuse_data["metrics"][m_idx]
        bfuse_num_opcodes     = bfuse_inst_per_opcode.num_instances()
        bfuse_opcodes         = bfuse_inst_per_opcode.correlation_ids()
        for i in range(bfuse_num_opcodes):
            op  = bfuse_opcodes.as_string(i)
            num = bfuse_inst_per_opcode.as_uint64(i)
            if op in bfuse_dict:
                bfuse_dict[op].append(num)
            else:
                bfuse_dict[op] = [num]

            # Append opcode name
            if op not in metrics_name:
                metrics_name.append(op)
        bfuse_list.append(bfuse_dict)

    # Preprocess datas
    datas     = {}
    print_max = 5
    for case, k1_dict, k2_dict, bfuse_dict in zip(cases, k1_list, k2_list, bfuse_list):
        print_cur = 0
        for m_name in metrics_name:
            k1_datas    = k1_dict.get(m_name)
            k2_datas    = k2_dict.get(m_name)
            bfuse_datas = bfuse_dict.get(m_name)

            if not k1_datas and not k2_datas:
                if print_cur < print_max:
                    logging.warning(f"Opcode \"{m_name}\" only exists in bfuse kernel. ({case[0]} x {case[1]})")
                    print_cur += 1
                    if print_cur == print_max:
                        logging.warning("Too many print in that case... skip")
                continue
            elif not bfuse_datas:
                if print_cur < print_max:
                    logging.warning(f"Opcode \"{m_name}\" doesn't exist in bfuse kernel. ({case[0]} x {case[1]})")
                    print_cur += 1
                    if print_cur == print_max:
                        logging.warning("Too many print in that case... skip")
                continue
            else:
                k1_mean    = np.mean(k1_datas) if k1_datas else 0
                k2_mean    = np.mean(k2_datas) if k2_datas else 0
                bfuse_mean = np.mean(bfuse_datas)
                diff_data  = bfuse_mean / (k1_mean + k2_mean)
                if m_name in datas:
                    datas[m_name].append(diff_data)
                else:
                    datas[m_name] = [diff_data]
    
    # Figure settings
    font = {"fontsize": 5, "fontweight": "bold"}
    y    = [i for i in range(len(metrics_name))]
    x    = []
    for m_name in metrics_name:
        data_list = datas.get(m_name)
        if data_list:
            value = np.mean(data_list)
        else:
            value = 0
        x.append(value)

    # Draw figure
    plt.title(figure_name, fontdict=font)
    plt.barh(y, x, height=0.5)
    plt.yticks(y, metrics_name)
    for i in y:
        plt.axhline(i + 0.5, color="gray", linestyle="--", linewidth=0.5)
    if xlim:
        plt.xlim(xlim)
    plt.axvline(1, color="red", linestyle="--", linewidth=0.5)
    plt.tick_params(axis="x", labelsize=5)
    plt.tick_params(axis="y", labelsize=5)

    # Save figure
    file_path = os.path.join(output_path, f"{figure_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close()

    # Print statistics
    print(f"==== {figure_name.upper()} ====")
    print("---------------------------------------")
    print(f"Total num: {len(perf_list)}")
    print(f"Perf max:  {max(perf_list):.2f}x")
    print(f"Perf min:  {min(perf_list):.2f}x")
    print(f"Perf avr:  {np.mean(perf_list):.2f}x")
    print(f"Perf std:  {np.std(perf_list):.2f}")
    print("---------------------------------------")
    for idx, m_name in enumerate(metrics_name):
        print(f"{m_name:10s}: {np.mean(x[idx]):.2f}x")
    print("---------------------------------------")
#-----------------------------------------------------------------------------------------------
def draw_figure(cases, kernel1, kernel2, parallel, hfuse, bfuse, output_path):
    mIdx = 0

    #################################################
    # GPU Speed Of Light Throughput
    #################################################

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
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = [0, 1, 2, 3, 4, 8, 9]
    elapsed_list     = []
    active_list      = []
    sum_list         = [5, 6, 7]
    low_is_best_list = [5, 6, 7]
    draw_scatter("GPU_Speed_Of_Light_Throughput", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Compute Workload Analysis
    #################################################

    metrics_name = [
        "Executed Ipc Elapsed [inst/cycle]",
        "Executed Ipc Active  [inst/cycle]",
        "Issued Ipc Active    [inst/cycle]",
        "SM Busy              [%]",
        "Issue Slots Busy     [%]",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = [0, 1, 2, 3, 4]
    elapsed_list     = []
    active_list      = []
    sum_list         = []
    low_is_best_list = []
    draw_scatter("Compute_Workload_Analysis", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Memory Workload Analysis
    #################################################

    metrics_name = [
        "Memory Throughput           [Gbyte/second]",
        "L1/TEX Hit Rate             [%]",
        "L2 Hit Rate                 [%]",
        "L2 Compression Success Rate [%]",
        "Mem Busy                    [%]",
        "Max Bandwidth               [%]",
        "Mem Pipes Busy              [%]",
        "L2 Compression Ratio",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = [0, 1, 2, 3, 4, 5, 6, 7]
    elapsed_list     = []
    active_list      = []
    sum_list         = []
    low_is_best_list = [7]
    draw_scatter("Memory_Workload_Analysis", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Scheduler Statistics
    #################################################

    metrics_name = [
        "Active Warps Per Scheduler   [warp]",
        "Eligible Warps Per Scheduler [warp]",
        "Issued Warp Per Scheduler",
        "No Eligible                  [%]",
        "One or More Eligible         [%]",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = []
    elapsed_list     = []
    active_list      = []
    sum_list         = []
    low_is_best_list = [3]
    draw_scatter("Scheduler_Statistics", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Warp State Statistics
    #################################################

    metrics_name = [
        "Warp Cycles Per Issued Instruction       [cycle]",
        "Warp Cycles Per Executed Instruction     [cycle]",
        "Avg. Active Threads Per Warp",
        "Avg. Not Predicated Off Threads Per Warp",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = [0, 1, 2, 3] # TODO: need weighted average using occupancy
    elapsed_list     = []
    active_list      = []
    sum_list         = []
    low_is_best_list = [0, 1, 3]
    draw_scatter("Warp_State_Statistics", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Instruction Statistics
    #################################################

    metrics_name = [
        "Executed Instructions                    [inst]",
        "Issued Instructions                      [inst]",
        "Avg. Executed Instructions Per Scheduler [inst]",
        "Avg. Issued Instructions Per Scheduler   [inst]",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = []
    elapsed_list     = []
    active_list      = []
    sum_list         = [0, 1, 2, 3]
    low_is_best_list = [0, 1, 2, 3]
    draw_scatter("Instruction_Statistics", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Breakdown: Executed Instruction Mix
    #################################################

    data_pack   = list(zip(cases, kernel1, kernel2, bfuse))
    metrics_idx = mIdx
    # draw_inst_breakdwon_hist("Breakdown-Executed_Instruction_Mix", output_path, data_pack, metrics_idx)
    mIdx += 1

    #################################################
    # Breakdown: Warp State (All Cycles)
    #################################################

    metrics_name = [
        "Stall Barrier",
        "Stall Branch Resolving",
        "Stall Dispatch Stall",
        "Stall Drain",
        "Stall IMC Miss",
        "Stall LG Throttle",
        "Stall Long Scoreboard",
        "Stall Math Pipe Throttle",
        "Stall Membar",
        "Stall MIO Throttle",
        "Stall Misc",
        "Stall No Instruction",
        "Stall Not Selected",
        "Selected",
        "Stall Short Scoreboard",
        "Stall Sleeping",
        "Stall Tex Throttle",
        "Stall Wait",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    elapsed_list     = []
    active_list      = []
    sum_list         = []
    low_is_best_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    draw_scatter("Breakdown-Warp_State_(All_Cycles)", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    draw_stall_breakdown_scatter("Breakdown-Kernel-Warp_State_(All_Cycles)", output_path, metrics_name, metrics_idx, data_pack, low_is_best_list)
    mIdx += len(metrics_name)

    #################################################
    # Breakdown: Pipe Utilization (% of peak instructions executed)
    #################################################

    metrics_name = [
        "LSU",
        "FMA",
        "ALU",
        "ADU",
        "Uniform",
        "CBU",
        "FMA (FP16)",
        "FP64 (DMMA)",
        "FP64 (FP64)",
        "TEX",
        "Tensor (FP)",
        "Tensor (INT)",
        "XU",
    ]
    metrics_idx      = [i for i in range(mIdx, mIdx + len(metrics_name))]
    data_pack        = list(zip(cases, kernel1, kernel2, bfuse))
    second_list      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elapsed_list     = []
    active_list      = []
    sum_list         = []
    low_is_best_list = []
    draw_scatter("Breakdown-Pipe_Utilization_(percent_of_peak_instructions_executed)-scatter", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    draw_compute_breakdown_scatter("Breakdown-Kernel-Pipe_Utilization_(percent_of_peak_instructions_executed)-scatter", output_path, metrics_name, metrics_idx, data_pack, second_list, elapsed_list, active_list, sum_list, low_is_best_list)
    mIdx += len(metrics_name)
#-----------------------------------------------------------------------------------------------
def draw_speedup_figure(figure_name, cases, kernel1, kernel2, parallel, hfuse, bfuse, output_path):
    materials_list = []
    parallel_list  = []
    hfuse_list     = []
    bfuse_list     = []

    for idx, [bi, ci] in enumerate(cases):
            materials_list.append(f"Kernel {bi} x Kernel {ci}")

            serial_exec   = kernel1[idx]["exec"] + kernel2[idx]["exec"]
            parallel_exec = serial_exec / parallel[idx]["exec"]
            hfuse_exec    = serial_exec / hfuse[idx]["exec"]
            bfuse_exec    = serial_exec / bfuse[idx]["exec"]
            
            parallel_list.append(parallel_exec)
            hfuse_list.append(hfuse_exec)
            bfuse_list.append(bfuse_exec)

    # Draw
    # plt.figure(figsize=(15, 15))

    idx = 0
    col_len = 21
    while idx < len(materials_list):
        idx_start = idx
        idx_end   = idx + col_len
        if idx_end > len(materials_list):
            idx_end = len(materials_list)

        index  = np.arange(idx_end - idx_start)
        axes_m = materials_list[idx_start:idx_end]
        axes_p = parallel_list[idx_start:idx_end]
        axes_h = hfuse_list[idx_start:idx_end]
        axes_b = bfuse_list[idx_start:idx_end]

        plt.subplot(math.ceil(len(materials_list) / col_len), 1, math.ceil(idx / col_len) + 1)

        plt.bar(index - 0.2, axes_p, width=0.2, label='parallel')
        plt.bar(index      , axes_h, width=0.2, label='hfuse')
        plt.bar(index + 0.2, axes_b, width=0.2, label='bfuse')

        # Draw baseline
        plt.axhline(y=1, color="r", linewidth=0.5, linestyle="--")

        # # Set ticks
        # plt.xticks(index, axes_m, fontsize=3)

        # Set limitation of y axis
        plt.ylim(0, 2)

        plt.ylabel('Speed up', fontsize=3)
        plt.legend(ncol=1, loc="upper left", fontsize=2)

        idx = idx_end
        if idx == len(materials_list):
            # 
            plt.axvline(x=index[-1] + 1, color="gray", linewidth=0.5, linestyle="--")

            # Geomean
            plt.bar([index[-1] + 2 - 0.2], gmean(parallel_list), width=0.2, label='parallel')
            plt.bar([index[-1] + 2      ], gmean(hfuse_list), width=0.2, label='hfuse')
            plt.bar([index[-1] + 2 + 0.2], gmean(bfuse_list), width=0.2, label='bfuse')
            break

    # Save figure
    file_path = os.path.join(output_path, f"{figure_name}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=500)
    plt.close()

    print(f"==== {figure_name.upper()} ====")
    print("---------------------------------------")
    print(f"parallel | max: {np.max(parallel_list)}, min: {np.min(parallel_list)}, avg: {gmean(parallel_list)}, std: {np.std(parallel_list)}")
    print(f"hfuse    | max: {np.max(hfuse_list)}, min: {np.min(hfuse_list)}, avg: {gmean(hfuse_list)}, std: {np.std(hfuse_list)}")
    print(f"bfuse    | max: {np.max(bfuse_list)}, min: {np.min(bfuse_list)}, avg: {gmean(bfuse_list)}, std: {np.std(bfuse_list)}")
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
    parser.add_argument("-i", action="store_true", default=False, dest="inst_cond",
                        help="options for selecting datas with condition of instruction num")
    parser.add_argument("-p", action="store_true", default=False, dest="pipe_cond",
                        help="options for selecting datas with condition of pipeline utilization")
    parser.add_argument("-o", action="store", default=".", dest="output",
                        help="output path of figure")
    parser.add_argument("file", action="store", help="path of generated test-suite")

    # Get arguments
    args        = parser.parse_args()
    show        = args.show
    inst_cond   = args.inst_cond
    pipe_cond   = args.pipe_cond
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
    cases, kernel1, kernel2, parallel, hfuse, bfuse = collect_datas_with_condition(yaml_info, exec_path, metrics_path, lambda x: True, inst_cond, pipe_cond)
    # cases, kernel1, kernel2, parallel, hfuse, bfuse = collect_datas_with_condition(yaml_info, exec_path, metrics_path, lambda x: x >= 1.1, inst_cond, pipe_cond)
    
    # Draw figure
    draw_figure(cases, kernel1, kernel2, parallel, hfuse, bfuse, output_path)

    # Draw speedup figure
    draw_speedup_figure("Speed_up", cases, kernel1, kernel2, parallel, hfuse, bfuse, output_path)