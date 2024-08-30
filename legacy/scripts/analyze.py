#!/usr/bin/python3

import os, shutil
import logging
import yaml, csv
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean, norm
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
        logging.error("Number of fusion sets are only 2.")
        exit(1)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])

    kernel1_metrics = []
    kernel2_metrics = []
    hfuse_metrics   = []
    bfuse_metrics   = []
    metrics_list    = ["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", # 0
                       "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
                       "l1tex__t_sector_hit_rate.pct", # 7
                       "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", # 8
                       "smsp__average_warp_latency_per_inst_issued.ratio", # 9
                       "l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_hit.sum",
                       "l1tex__t_sectors_pipe_tex_mem_texture_lookup_hit.sum", # 21
                       "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
                       "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio", # 32
                       "sm__inst_issued.avg.per_cycle_active", # 33
                       "sm__cycles_active.avg", # 34
                       "smsp__inst_issued.sum", # 35
                       "smsp__average_warp_latency_per_inst_issued.ratio", # 36
                       "smsp__average_warps_active_per_inst_executed.ratio" # 37
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

    for i0 in range(kernel1_size):
        temp_list = []
        for i1 in range(kernel2_size):
            if i0 <= i1:
                report_path = os.path.join("metrics", f"3_{i0}_{i1}.ncu-rep")
            else:
                report_path = os.path.join("metrics", f"3_{i1}_{i0}.ncu-rep")
            ncu_context = ncu_report.load_report(report_path)

            ncu_range  = ncu_context.range_by_idx(0)
            ncu_action = ncu_range.action_by_idx(0)

            metrics = []
            for m in metrics_list:
                metrics.append(ncu_action[m])
            temp_list.append(metrics)
        hfuse_metrics.append(temp_list)

    for i0 in range(kernel1_size):
        temp_list = []
        for i1 in range(kernel2_size):
            report_path = os.path.join("metrics", f"4_{i0}_{i1}.ncu-rep")
            ncu_context = ncu_report.load_report(report_path)

            ncu_range  = ncu_context.range_by_idx(0)
            ncu_action = ncu_range.action_by_idx(0)

            metrics = []
            for m in metrics_list:
                metrics.append(ncu_action[m])
            temp_list.append(metrics)
        bfuse_metrics.append(temp_list)

    return metrics_list, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics
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
def analyze_cond(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):
    paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

    metric_statistics = [[] for _ in range(9)]
    bfuse_statistics  = [[] for _ in range(9)]
    print(f"================================================== {name.upper()} CASES ==================================================")
    for idx, [bi, ci, bfuse_exec] in enumerate(analysis):
        if idx >= 5:
            print("")
            print("Too many cases...")
            break

        # Kernel1
        kernel1_math_throttle = f"{kernel1_metrics[bi][0].value():.2f}"
        kernel1_lg_throttle   = f"{kernel1_metrics[bi][1].value():.2f}"
        kernel1_mio_throttle  = f"{kernel1_metrics[bi][2].value():.2f}"
        kernel1_tex_throttle  = f"{kernel1_metrics[bi][3].value():.2f}"
        kernel1_math_depend   = f"{kernel1_metrics[bi][4].value():.2f}"
        kernel1_lg_depend     = f"{kernel1_metrics[bi][5].value():.2f}"
        kernel1_mio_depend    = f"{kernel1_metrics[bi][6].value():.2f}"
        kernel1_L1Tex_hit     = f"{kernel1_metrics[bi][7].value():.2f}"
        kernel1_bank_conflict = f"{kernel1_metrics[bi][8].as_uint64()}"

        # Kernel2
        kernel2_math_throttle = f"{kernel2_metrics[ci][0].value():.2f}"
        kernel2_lg_throttle   = f"{kernel2_metrics[ci][1].value():.2f}"
        kernel2_mio_throttle  = f"{kernel2_metrics[ci][2].value():.2f}"
        kernel2_tex_throttle  = f"{kernel2_metrics[ci][3].value():.2f}"
        kernel2_math_depend   = f"{kernel2_metrics[ci][4].value():.2f}"
        kernel2_lg_depend     = f"{kernel2_metrics[ci][5].value():.2f}"
        kernel2_mio_depend    = f"{kernel2_metrics[ci][6].value():.2f}"
        kernel2_L1Tex_hit     = f"{kernel2_metrics[ci][7].value():.2f}"
        kernel2_bank_conflict = f"{kernel2_metrics[ci][8].as_uint64()}"

        # HFuse
        if bi <= ci:
            hfuse_math_throttle = f"{hfuse_metrics[bi][ci][0].value():.2f}"
            hfuse_lg_throttle   = f"{hfuse_metrics[bi][ci][1].value():.2f}"
            hfuse_mio_throttle  = f"{hfuse_metrics[bi][ci][2].value():.2f}"
            hfuse_tex_throttle  = f"{hfuse_metrics[bi][ci][3].value():.2f}"
            hfuse_math_depend   = f"{hfuse_metrics[bi][ci][4].value():.2f}"
            hfuse_lg_depend     = f"{hfuse_metrics[bi][ci][5].value():.2f}"
            hfuse_mio_depend    = f"{hfuse_metrics[bi][ci][6].value():.2f}"
            hfuse_L1Tex_hit     = f"{hfuse_metrics[bi][ci][7].value():.2f}"
            hfuse_bank_conflict = f"{hfuse_metrics[bi][ci][8].as_uint64()}"
        else:
            hfuse_math_throttle = f"{hfuse_metrics[ci][bi][0].value():.2f}"
            hfuse_lg_throttle   = f"{hfuse_metrics[ci][bi][1].value():.2f}"
            hfuse_mio_throttle  = f"{hfuse_metrics[ci][bi][2].value():.2f}"
            hfuse_tex_throttle  = f"{hfuse_metrics[ci][bi][3].value():.2f}"
            hfuse_math_depend   = f"{hfuse_metrics[ci][bi][4].value():.2f}"
            hfuse_lg_depend     = f"{hfuse_metrics[ci][bi][5].value():.2f}"
            hfuse_mio_depend    = f"{hfuse_metrics[ci][bi][6].value():.2f}"
            hfuse_L1Tex_hit     = f"{hfuse_metrics[ci][bi][7].value():.2f}"
            hfuse_bank_conflict = f"{hfuse_metrics[ci][bi][8].as_uint64()}"

        # BFuse
        bfuse_math_throttle = f"{bfuse_metrics[bi][ci][0].value():.2f}"
        bfuse_lg_throttle   = f"{bfuse_metrics[bi][ci][1].value():.2f}"
        bfuse_mio_throttle  = f"{bfuse_metrics[bi][ci][2].value():.2f}"
        bfuse_tex_throttle  = f"{bfuse_metrics[bi][ci][3].value():.2f}"
        bfuse_math_depend   = f"{bfuse_metrics[bi][ci][4].value():.2f}"
        bfuse_lg_depend     = f"{bfuse_metrics[bi][ci][5].value():.2f}"
        bfuse_mio_depend    = f"{bfuse_metrics[bi][ci][6].value():.2f}"
        bfuse_L1Tex_hit     = f"{bfuse_metrics[bi][ci][7].value():.2f}"
        bfuse_bank_conflict = f"{bfuse_metrics[bi][ci][8].as_uint64()}"

        # Diff
        diff_metrics = []
        for idx in range(9):
            bi_value    = kernel1_metrics[bi][idx].value()
            ci_value    = kernel2_metrics[ci][idx].value()
            bfuse_value = bfuse_metrics[bi][ci][idx].value()

            if idx >= 0 and idx < 7:
                if bi_value == 0 and ci_value == 0:
                    diff = "NaN"
                else:
                    bi_inst = kernel1_metrics[bi][9].value()
                    ci_inst = kernel2_metrics[ci][9].value()
                    result  = bfuse_value / ((bi_value * bi_inst + ci_value * ci_inst) / (bi_inst + ci_inst))
                    diff    = f"{result:.2f}"
                    metric_statistics[idx].append(result)
            if idx == 7:
                if bi_value == 0 and ci_value == 0:
                    diff = "NaN"
                elif bi_value == 0:
                    diff2 = bfuse_value / ci_value
                    diff  = f"{diff2:.2f}"
                    metric_statistics[idx].append(diff2)
                elif ci_value == 0:
                    diff1 = bfuse_value / bi_value
                    diff  = f"{diff1:.2f}"
                    metric_statistics[idx].append(diff1)
                else:
                    result = bfuse_value / ((bi_value + ci_value) / 2)
                    diff = f"{result:.2f}"
                    metric_statistics[idx].append(result)
            elif idx == 8:
                if bi_value + ci_value == 0:
                    diff = "NaN"
                else:
                    result = bfuse_value / (bi_value + ci_value)
                    diff   = f"{result:.2f}"
                    metric_statistics[idx].append(result)
            else:
                pass
            bfuse_statistics[idx].append(bfuse_value)
            diff_metrics.append(diff)

        # Unit
        unit_math_throttle = f"{kernel1_metrics[bi][0].unit()}"
        unit_lg_throttle   = f"{kernel1_metrics[bi][1].unit()}"
        unit_mio_throttle  = f"{kernel1_metrics[bi][2].unit()}"
        unit_tex_throttle  = f"{kernel1_metrics[bi][3].unit()}"
        unit_math_depend   = f"{kernel1_metrics[bi][4].unit()}"
        unit_lg_depend     = f"{kernel1_metrics[bi][5].unit()}"
        unit_mio_depend    = f"{kernel1_metrics[bi][6].unit()}"
        unit_L1Tex_hit     = f"{kernel1_metrics[bi][7].unit()}"
        unit_bank_conflict = f"{kernel1_metrics[bi][8].unit()}"

        # Print
        print(f"case <conv2d_{bi} x conv2d_{ci}>:")
        print("==== STALL ====")
        print(f" - MATH throttle stall: {kernel1_math_throttle:>5s}/{kernel2_math_throttle:>5s}/{hfuse_math_throttle:>5s}/{bfuse_math_throttle:>5s}  ({diff_metrics[0]:>5s}x) ({unit_math_throttle})")
        print(f" - LG throttle stall:   {kernel1_lg_throttle:>5s}/{kernel2_lg_throttle:>5s}/{hfuse_lg_throttle:>5s}/{bfuse_lg_throttle:>5s}  ({diff_metrics[1]:>5s}x) ({unit_lg_throttle})")
        print(f" - MIO throttle stall:  {kernel1_mio_throttle:>5s}/{kernel2_mio_throttle:>5s}/{hfuse_mio_throttle:>5s}/{bfuse_mio_throttle:>5s}  ({diff_metrics[2]:>5s}x) ({unit_mio_throttle})")
        print(f" - TEX throttle stall:  {kernel1_tex_throttle:>5s}/{kernel2_tex_throttle:>5s}/{hfuse_tex_throttle:>5s}/{bfuse_tex_throttle:>5s}  ({diff_metrics[3]:>5s}x) ({unit_tex_throttle})")
        print(f" - MATH depend stall:   {kernel1_math_depend:>5s}/{kernel2_math_depend:>5s}/{hfuse_math_depend:>5s}/{bfuse_math_depend:>5s}  ({diff_metrics[4]:>5s}x) ({unit_math_depend})")
        print(f" - LG depend stall:     {kernel1_lg_depend:>5s}/{kernel2_lg_depend:>5s}/{hfuse_lg_depend:>5s}/{bfuse_lg_depend:>5s}  ({diff_metrics[5]:>5s}x) ({unit_lg_depend})")
        print(f" - MIO depend stall:    {kernel1_mio_depend:>5s}/{kernel2_mio_depend:>5s}/{hfuse_mio_depend:>5s}/{bfuse_mio_depend:>5s}  ({diff_metrics[6]:>5s}x) ({unit_mio_depend})")
        print("==== MEMORY ====")
        print(f" - L1/Tex Cache Hit:            {kernel1_L1Tex_hit:>13s}/{kernel2_L1Tex_hit:>13s}/{hfuse_L1Tex_hit:>13s}/{bfuse_L1Tex_hit:>13s}  ({diff_metrics[7]:>5s}x) ({unit_L1Tex_hit})")
        print(f" - Shared Memory Bank Conflict: {kernel1_bank_conflict:>13s}/{kernel2_bank_conflict:>13s}/{hfuse_bank_conflict:>13s}/{bfuse_bank_conflict:>13s}  ({diff_metrics[8]:>5s}x) ({unit_bank_conflict})")
        print("==== Performance ====")
        print(f" - Improvement: {bfuse_exec:.2f}x")

    # Print statistics
    max_perf = 0
    min_perf = float("inf")
    for _, _, bfuse_exec in analysis:
        if bfuse_exec > max_perf:
            max_perf = bfuse_exec
        if bfuse_exec < min_perf:
            min_perf = bfuse_exec

    bfuse_math_throttle_mean = f"{np.mean(bfuse_statistics[0]):.2f}"
    bfuse_lg_throttle_mean   = f"{np.mean(bfuse_statistics[1]):.2f}"
    bfuse_mio_throttle_mean  = f"{np.mean(bfuse_statistics[2]):.2f}"
    bfuse_tex_throttle_mean  = f"{np.mean(bfuse_statistics[3]):.2f}"
    bfuse_math_depend_mean   = f"{np.mean(bfuse_statistics[4]):.2f}"
    bfuse_lg_depend_mean     = f"{np.mean(bfuse_statistics[5]):.2f}"
    bfuse_mio_depend_mean    = f"{np.mean(bfuse_statistics[6]):.2f}"
    bfuse_L1Tex_hit_mean     = f"{np.mean(bfuse_statistics[7]):.2f}"
    bfuse_bank_conflict_mean = f"{np.mean(bfuse_statistics[8]):.2f}"

    math_throttle_mean = f"{np.mean(metric_statistics[0]):.2f}"
    lg_throttle_mean = f"{np.mean(metric_statistics[1]):.2f}"
    mio_throttle_mean = f"{np.mean(metric_statistics[2]):.2f}"
    tex_throttle_mean = f"{np.mean(metric_statistics[3]):.2f}"
    math_depend_mean = f"{np.mean(metric_statistics[4]):.2f}"
    lg_depend_mean = f"{np.mean(metric_statistics[5]):.2f}"
    mio_depend_mean = f"{np.mean(metric_statistics[6]):.2f}"
    L1Tex_hit_mean = f"{np.mean(metric_statistics[7]):.2f}"
    bank_conflict_mean = f"{np.mean(metric_statistics[8]):.2f}"

    print("-------- STATISTICS --------")
    print(f"Total num: {len(analysis)}")
    print(f"Max perf:  {max_perf:.2f}x")
    print(f"Min perf:  {min_perf:.2f}x")
    print("==== STALL ====")
    print(f" - MATH throttle stall: {bfuse_math_throttle_mean:>5s} ({math_throttle_mean:>5s}x)")
    print(f" - LG throttle stall:   {bfuse_lg_throttle_mean:>5s} ({lg_throttle_mean:>5s}x)")
    print(f" - MIO throttle stall:  {bfuse_mio_throttle_mean:>5s} ({mio_throttle_mean:>5s}x)")
    print(f" - TEX throttle stall:  {bfuse_tex_throttle_mean:>5s} ({tex_throttle_mean:>5s}x)")
    print(f" - MATH depend stall:   {bfuse_math_depend_mean:>5s} ({math_depend_mean:>5s}x)")
    print(f" - LG depend stall:     {bfuse_lg_depend_mean:>5s} ({lg_depend_mean:>5s}x)")
    print(f" - MIO depend stall:    {bfuse_mio_depend_mean:>5s} ({mio_depend_mean:>5s}x)")
    print("==== MEMORY ====")
    print(f" - L1/Tex Cache Hit:            {bfuse_L1Tex_hit_mean:>5s} ({L1Tex_hit_mean:>5s}x)")
    print(f" - Shared Memory Bank Conflict: {bfuse_bank_conflict_mean} ({bank_conflict_mean}x)")
#-----------------------------------------------------------------------------------------------
# def pattern_cond(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):
#     paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

#     metrics_name = ["math_throttle",
#                     "lg_throttle",
#                     "mio_throttle",
#                     "tex_throttle",
#                     "math_depend",
#                     "lg_depend",
#                     "mio_depend",
#                     "L1Tex_hit",
#                     "bank_conflict",
#                     ]

#     kernel1_list = []
#     kernel2_list = []
#     bfuse_list   = []
#     diff_list    = []
#     for bi, ci, bfuse_exec in analysis:
#         k1    = []
#         k2    = []
#         bfuse = []
#         diff  = []
#         for idx in range(len(metrics_name)):
#             k1_value    = kernel1_metrics[bi][idx].value()
#             k2_value    = kernel2_metrics[ci][idx].value()
#             bfuse_value = bfuse_metrics[bi][ci][idx].value()

#             k1.append(k1_value)
#             k2.append(k2_value)
#             bfuse.append(bfuse_value)

#             if k1_value == 0 and k2_value == 0:
#                 diff.append(float("nan"))
#                 continue

#             if idx >= 0 and idx < 7:
#                 k1_inst = kernel1_metrics[bi][9].value()
#                 k2_inst = kernel2_metrics[ci][9].value()
#                 # result  = bfuse_value / ((k1_value * k1_inst + k2_value * k2_inst) / (k1_inst + k2_inst))
#                 bfuse_inst = bfuse_metrics[bi][ci][9].value()
#                 k1_value    = np.sum([e.value() for i, e in enumerate(kernel1_metrics[bi][:4])])
#                 k2_value    = np.sum([e.value() for i, e in enumerate(kernel2_metrics[ci][:4])])
#                 bfuse_value = np.sum([e.value() for i, e in enumerate(bfuse_metrics[bi][ci][:4])])
#                 result  = (bfuse_value * bfuse_inst) / (k1_value * k1_inst + k2_value * k2_inst)
#             elif idx == 7:
#                 result = bfuse_value / ((k1_value + k2_value) / 2)
#             elif idx == 8:
#                 result = bfuse_value / (k1_value + k2_value)
#             else:
#                 logging.error("Unable to reach here..!")
#                 exit(1)

#             diff.append(result)
        
#         kernel1_list.append(k1)
#         kernel2_list.append(k2)
#         bfuse_list.append(bfuse)
#         diff_list.append(diff)

#     # Find pattern
#     for idx, name in enumerate(metrics_name[:1]):
#         candidate = [[] for _ in range(len(metrics_name))]
#         cases     = []
#         for i, e in enumerate(diff_list):
#             # # if e.index(min(e[:3] + [1] + e[4:7])) != idx:
#             # if e[idx] > 1.0:
#             #     continue

#             for j in range(len(metrics_name)):
#                 if e[j] == float("nan"):
#                     continue
#                 candidate[j].append(e[j])
#             cases.append(i)
            
#         print(f"--------<'{name.upper()}' BEST CASES>--------")
#         print(f"Total num: {len(cases)}/{len(diff_list)} ({len(cases) / len(diff_list) * 100:.2f}%)")
#         if len(cases) == 0:
#             print("==== NO CASES ====")
#             continue

#         perf_list = []
#         for c in cases:
#             perf_list.append(analysis[c][2])
#         mean_result = []
#         for cand in candidate:
#             mean_result.append(f"{np.mean(cand):.2f}")
#             # mean_result.append(f"{np.average(cand, weights=perf_list):.2f}")

#         # Print statistics
#         print(f"Perf max: {np.max(perf_list):.2f}x")
#         print(f"Perf min: {np.min(perf_list):.2f}x")
#         print(f"Perf avr: {np.mean(perf_list):.2f}x")
#         print(f"Perf std: {np.std(perf_list):.2f}")
#         print("==== STALL ====")
#         print(f" - MATH throttle stall: {mean_result[0]:>5s}x")
#         print(f" - LG throttle stall:   {mean_result[1]:>5s}x")
#         print(f" - MIO throttle stall:  {mean_result[2]:>5s}x")
#         print(f" - TEX throttle stall:  {mean_result[3]:>5s}x")
#         print(f" - MATH depend stall:   {mean_result[4]:>5s}x")
#         print(f" - LG depend stall:     {mean_result[5]:>5s}x")
#         print(f" - MIO depend stall:    {mean_result[6]:>5s}x")
#         print("==== MEMORY ====")
#         print(f" - L1/Tex Cache Hit:            {mean_result[7]:>5s}x")
#         print(f" - Shared Memory Bank Conflict: {mean_result[8]:>5s}x")

#         # Draw bfuse perf breakdown
#         title_font = {'fontsize': 5, 'fontweight': 'bold'}
#         # plt.title(f"{name}")
#         for i, cand in enumerate(candidate):
#             if i == 3: # Tex throttle
#                 continue

#             if i < 3:
#                 pi = i + 1
#             else:
#                 pi = i
#             plt.subplot(math.ceil(len(metrics_name) / 3), 3, pi)
#             plt.title(f"{metrics_name[i]}", fontdict=title_font)
#             if i == 7:
#                 plt.scatter(perf_list, cand, s=0.5**2, c="#FF7F00")
#             else:
#                 plt.scatter(perf_list, cand, s=0.5**2)
#             plt.axhline(1, color="red", linestyle="--", linewidth=0.5)
#             # if i == 8:
#             #     plt.ylim([0, 10])
#             # plt.ylim([0, 3])
#             plt.tick_params(axis='x', labelsize=5)
#             plt.tick_params(axis='y', labelsize=5)
        
#         plt.subplot(math.ceil(len(metrics_name) / 3), 3, len(metrics_name))
#         plt.title("Histogram", fontdict=title_font)
#         plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
#         plt.tick_params(axis='x', labelsize=5)
#         plt.tick_params(axis='y', labelsize=5)
#         plt.tight_layout()
#         # plt.savefig(f"figure/figure_1-{name}.png", dpi=500)
#         plt.savefig(f"figure/figure_1.png", dpi=500)
#         plt.close()

#         # Draw kernel1/2 breakdown
#         for i, m_name in enumerate(metrics_name):
#             if i == 3: # Tex throttle
#                 continue
            
#             data_1 = []
#             data_2 = []
#             diffs  = []
#             for c in cases:
#                 value_1 = kernel1_list[c][i]
#                 value_2 = kernel2_list[c][i]
#                 if value_1 > value_2:
#                     value_1, value_2 = value_2, value_1
#                 data_1.append(value_1)
#                 data_2.append(value_2)
#                 if value_1 + value_2 == 0:
#                     diffs.append(0)
#                 else:
#                     diffs.append(abs(value_1 - value_2))

#             if i < 3:
#                 pi = i + 1
#             else:
#                 pi = i
#             plt.subplot(math.ceil(len(metrics_name) / 3), 3, pi)
#             plt.title(f"{m_name}", fontdict=title_font)
#             # plt.scatter(perf_list, data_1, s=0.5**2)
#             # plt.scatter(perf_list, data_2, s=0.5**2, c="#FF7F00")
#             plt.scatter(perf_list, diffs, s=0.5**2)
#             plt.tick_params(axis='x', labelsize=5)
#             plt.tick_params(axis='y', labelsize=5)
        
#         plt.subplot(math.ceil(len(metrics_name) / 3), 3, len(metrics_name))
#         plt.title("Histogram", fontdict=title_font)
#         plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
#         plt.tick_params(axis='x', labelsize=5)
#         plt.tick_params(axis='y', labelsize=5)
#         plt.tight_layout()
#         # plt.savefig(f"figure/figure_2-{name}.png", dpi=500)
#         plt.savefig(f"figure/figure_2.png", dpi=500)
#         plt.close()
#-----------------------------------------------------------------------------------------------
def pattern_cond2(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):
    paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

    metrics_name = ["math_throttle",
                    "lg_throttle",
                    "mio_throttle",
                    "tex_throttle",
                    "math_depend",
                    "lg_depend",
                    "mio_depend",
                    "L1Tex_hit",
                    "bank_conflict",
                    "ipc",
                    "inst_num",
                    ]

    kernel1_list = []
    kernel2_list = []
    bfuse_list   = []
    diff_list    = []
    for bi, ci, bfuse_exec in analysis:
        k1    = []
        k2    = []
        bfuse = []
        diff  = []
        for idx in range(len(kernel1_metrics[0])):
            if idx < 9 or idx == 33 or idx == 35:
                k1_value    = kernel1_metrics[bi][idx].value()
                k2_value    = kernel2_metrics[ci][idx].value()
                bfuse_value = bfuse_metrics[bi][ci][idx].value()
            else:
                continue

            k1.append(k1_value)
            k2.append(k2_value)
            bfuse.append(bfuse_value)

        for idx in range(len(metrics_name)):
            if idx < 9:
                k1_value    = kernel1_metrics[bi][idx].value()
                k2_value    = kernel2_metrics[ci][idx].value()
                bfuse_value = bfuse_metrics[bi][ci][idx].value()
            elif idx == 9:
                k1_value    = kernel1_metrics[bi][33].value()
                k2_value    = kernel2_metrics[ci][33].value()
                bfuse_value = bfuse_metrics[bi][ci][33].value()
            elif idx == 10:
                k1_value    = kernel1_metrics[bi][35].value()
                k2_value    = kernel2_metrics[ci][35].value()
                bfuse_value = bfuse_metrics[bi][ci][35].value()
            else:
                logging.error("Unable to reach here..!")
                exit(1)

            if k1_value == 0 and k2_value == 0:
                diff.append(float("nan"))
                continue

            if idx >= 0 and idx < 4:
                # k1_inst = kernel1_metrics[bi][9].value()
                # k2_inst = kernel2_metrics[ci][9].value()
                # bfuse_inst = bfuse_metrics[bi][ci][9].value()
                # k1_value    = np.sum([e.value() for e in kernel1_metrics[bi][:4]])
                # k2_value    = np.sum([e.value() for e in kernel2_metrics[ci][:4]])
                # bfuse_value = np.sum([e.value() for e in bfuse_metrics[bi][ci][:4]])
                # k1_value    = kernel1_metrics[bi][idx].value()
                # k2_value    = kernel2_metrics[ci][idx].value()
                # bfuse_value = bfuse_metrics[bi][ci][idx].value()
                # result  = (bfuse_value * bfuse_inst) / (k1_value * k1_inst + k2_value * k2_inst)
                k1_value    = kernel1_metrics[bi][36].value()
                k2_value    = kernel2_metrics[ci][36].value()
                bfuse_value = bfuse_metrics[bi][ci][36].value()
                result = bfuse_value / ((k1_value + k2_value) / 2)
            elif idx >= 4 and idx < 7:
                # k1_inst = kernel1_metrics[bi][9].value()
                # k2_inst = kernel2_metrics[ci][9].value()
                # bfuse_inst = bfuse_metrics[bi][ci][9].value()
                # k1_value    = np.sum([e.value() for e in kernel1_metrics[bi][4:7]])
                # k2_value    = np.sum([e.value() for e in kernel2_metrics[ci][4:7]])
                # bfuse_value = np.sum([e.value() for e in bfuse_metrics[bi][ci][4:7]])
                # k1_value    = kernel1_metrics[bi][idx].value()
                # k2_value    = kernel2_metrics[ci][idx].value()
                # bfuse_value = bfuse_metrics[bi][ci][idx].value()
                # result  = (bfuse_value * bfuse_inst) / (k1_value * k1_inst + k2_value * k2_inst)
                k1_value    = kernel1_metrics[bi][37].value()
                k2_value    = kernel2_metrics[ci][37].value()
                bfuse_value = bfuse_metrics[bi][ci][37].value()
                result = bfuse_value / ((k1_value + k2_value) / 2)
            elif idx == 7:
                # k1_value    = np.sum([e.value() for e in kernel1_metrics[bi][10:22]])
                # k2_value    = np.sum([e.value() for e in kernel2_metrics[ci][10:22]])
                # bfuse_value = np.sum([e.value() for e in bfuse_metrics[bi][ci][10:22]])
                k1_value    = kernel1_metrics[bi][7].value()
                k2_value    = kernel2_metrics[ci][7].value()
                bfuse_value = bfuse_metrics[bi][ci][7].value()
                result = bfuse_value / ((k1_value + k2_value) / 2)
            elif idx == 8:
                k1_value    = kernel1_metrics[bi][8].value()
                k2_value    = kernel2_metrics[ci][8].value()
                bfuse_value = bfuse_metrics[bi][ci][8].value()
                result = bfuse_value / (k1_value + k2_value)
            elif idx == 9:
                # k1_cycle    = kernel1_metrics[bi][34].value()
                # k2_cycle    = kernel2_metrics[ci][34].value()
                # bfuse_cycle = bfuse_metrics[bi][ci][34].value()
                # result = (bfuse_value * bfuse_cycle) / (k1_value * k1_cycle + k2_value * k2_cycle)
                k1_value    = kernel1_metrics[bi][33].value()
                k2_value    = kernel2_metrics[ci][33].value()
                bfuse_value = bfuse_metrics[bi][ci][33].value()
                result = bfuse_value / ((k1_value + k2_value) / 2)
            elif idx == 10:
                k1_value    = kernel1_metrics[bi][35].value()
                k2_value    = kernel2_metrics[ci][35].value()
                bfuse_value = bfuse_metrics[bi][ci][35].value()
                result = bfuse_value / (k1_value + k2_value)
                # if result > 2.0:
                #     print(f"case: {bi} x {ci}")
            else:
                logging.error("Unable to reach here..!")
                exit(1)

            diff.append(result)
        
        kernel1_list.append(k1)
        kernel2_list.append(k2)
        bfuse_list.append(bfuse)
        diff_list.append(diff)

    # Find pattern
    for idx, name in enumerate(metrics_name[:1]):
        candidate = [[] for _ in range(len(metrics_name))]
        cases     = []
        for i, e in enumerate(diff_list):
            # # if e.index(min(e[:3] + [1] + e[4:7])) != idx:
            # if e[idx] > 1.0:
            #     continue
            # if e[10] >= 1.05 or e[10] < 0.95:
            #     continue

            for j in range(len(metrics_name)):
                if e[j] == float("nan"):
                    continue
                candidate[j].append(e[j])
            cases.append(i)
            
        print(f"--------<'{name.upper()}' BEST CASES>--------")
        print(f"Total num: {len(cases)}/{len(diff_list)} ({len(cases) / len(diff_list) * 100:.2f}%)")
        if len(cases) == 0:
            print("==== NO CASES ====")
            continue

        perf_list = []
        for c in cases:
            perf_list.append(analysis[c][2])
        mean_result = []
        for cand in candidate:
            mean_result.append(f"{np.mean(cand):.2f}")
            # mean_result.append(f"{np.average(cand, weights=perf_list):.2f}")

        # Print statistics
        print(f"Perf max: {np.max(perf_list):.2f}x")
        print(f"Perf min: {np.min(perf_list):.2f}x")
        print(f"Perf avr: {np.mean(perf_list):.2f}x")
        print(f"Perf std: {np.std(perf_list):.2f}")
        print("==== STALL ====")
        print(f" - MATH throttle stall: {mean_result[0]:>5s}x")
        print(f" - LG throttle stall:   {mean_result[1]:>5s}x")
        print(f" - MIO throttle stall:  {mean_result[2]:>5s}x")
        print(f" - TEX throttle stall:  {mean_result[3]:>5s}x")
        print(f" - MATH depend stall:   {mean_result[4]:>5s}x")
        print(f" - LG depend stall:     {mean_result[5]:>5s}x")
        print(f" - MIO depend stall:    {mean_result[6]:>5s}x")
        print("==== MEMORY ====")
        print(f" - L1/Tex Cache Hit:            {mean_result[7]:>5s}x")
        print(f" - Shared Memory Bank Conflict: {mean_result[8]:>5s}x")

        # Draw bfuse perf breakdown
        title_font = {'fontsize': 5, 'fontweight': 'bold'}
        # plt.title(f"{name}")
        for i, cand in enumerate(candidate):
            if i == 3: # Tex throttle
                continue

            if i < 3:
                pi = i + 1
            else:
                pi = i
            plt.subplot(math.ceil(len(metrics_name) / 3), 3, pi)
            plt.title(f"{metrics_name[i]}", fontdict=title_font)
            if i == 7:
                plt.scatter(perf_list, cand, s=0.5**2, c="#FF7F00")
            else:
                plt.scatter(perf_list, cand, s=0.5**2)
            plt.axhline(1, color="red", linestyle="--", linewidth=0.5)
            # if i == 8:
            #     plt.ylim([0, 10])
            # plt.ylim([0, 3])
            plt.tick_params(axis='x', labelsize=5)
            plt.tick_params(axis='y', labelsize=5)
        
        plt.subplot(math.ceil(len(metrics_name) / 3), 3, len(metrics_name))
        plt.title("Histogram", fontdict=title_font)
        plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
        plt.tick_params(axis='x', labelsize=5)
        plt.tick_params(axis='y', labelsize=5)
        plt.tight_layout()
        # plt.savefig(f"figure/figure_1-{name}.png", dpi=500)
        plt.savefig(f"figure/figure_1.png", dpi=500)
        plt.close()

        # Draw kernel1/2 breakdown
        for i, m_name in enumerate(metrics_name):
            if i == 3: # Tex throttle
                continue

            diffs     = []
            diff_base = []
            for c in cases:
                if i >= 0 and i < 4:
                    k1_inst = kernel1_list[c][9]
                    k2_inst = kernel2_list[c][9]
                    # k1_total = np.sum([e for e in (kernel1_list[c][:4] + kernel1_list[c][4:7] + kernel1_list[c][22:33])])
                    # k2_total = np.sum([e for e in (kernel2_list[c][:4] + kernel2_list[c][4:7] + kernel2_list[c][22:33])])
                    # value_1 = np.sum([e for e in kernel1_list[c][:4]]) / k1_total * 100
                    # value_2 = np.sum([e for e in kernel2_list[c][:4]]) / k2_total * 100
                    # result  = abs(value_1 - value_2)
                    result = 0
                    for k in range(4):
                        value_1 = kernel1_list[c][k]
                        value_2 = kernel2_list[c][k]
                        result += abs(value_1 - value_2)
                    # value_1 = kernel1_list[c][i]
                    # value_2 = kernel2_list[c][i]
                    # result  = abs(value_1 * k1_inst - value_2 * k2_inst)
                    diffs.append(result)

                    value_1 = np.sum([e for e in kernel1_list[c][:4]])
                    value_2 = np.sum([e for e in kernel2_list[c][:4]])
                    b_inst  = bfuse_list[c][9]
                    # value_b = bfuse_list[c][i]
                    value_b = np.sum([e for e in bfuse_list[c][:4]])
                    result = (value_b * b_inst) / (value_1 * k1_inst + value_2 * k2_inst)
                    diff_base.append(result)
                elif i >= 4 and i < 7:
                    k1_inst = kernel1_list[c][9]
                    k2_inst = kernel2_list[c][9]
                    # k1_total = np.sum([e for e in (kernel1_list[c][:4] + kernel1_list[c][4:7] + kernel1_list[c][22:33])])
                    # k2_total = np.sum([e for e in (kernel2_list[c][:4] + kernel2_list[c][4:7] + kernel2_list[c][22:33])])
                    # value_1 = np.sum([e for e in kernel1_list[c][4:7]]) / k1_total * 100
                    # value_2 = np.sum([e for e in kernel2_list[c][4:7]]) / k2_total * 100
                    # result  = abs(value_1 - value_2)
                    result = 0
                    for k in range(4, 7):
                        value_1 = kernel1_list[c][k]
                        value_2 = kernel2_list[c][k]
                        result += abs(value_1 - value_2)
                    # value_1 = kernel1_list[c][i]
                    # value_2 = kernel2_list[c][i]
                    # result  = abs(value_1 * k1_inst - value_2 * k2_inst)
                    diffs.append(result)

                    value_1 = np.sum([e for e in kernel1_list[c][4:7]])
                    value_2 = np.sum([e for e in kernel2_list[c][4:7]])
                    b_inst  = bfuse_list[c][9]
                    # value_b = bfuse_list[c][i]
                    value_b = np.sum([e for e in bfuse_list[c][4:7]])
                    result = (value_b * b_inst) / (value_1 * k1_inst + value_2 * k2_inst)
                    diff_base.append(result)
                elif i == 7:
                    # value_1 = np.sum([e for e in kernel1_list[c][10:22]])
                    # value_2 = np.sum([e for e in kernel2_list[c][10:22]])
                    value_1 = kernel1_list[c][7]
                    value_2 = kernel2_list[c][7]
                    result  = abs(value_1 - value_2)
                    diffs.append(result)
                    diff_base.append(diff_list[c][7]) # temp
                elif i == 8:
                    value_1 = kernel1_list[c][8]
                    value_1 = kernel2_list[c][8]
                    result  = abs(value_1 - value_2)
                    diffs.append(result)
                    diff_base.append(diff_list[c][8]) # temp
                else:
                    logging.error("Unable to reach here..!")
                    exit(1)

            if i < 3:
                pi = i + 1
            else:
                pi = i
            plt.subplot(math.ceil(len(metrics_name) / 3), 3, pi)
            plt.title(f"{m_name}", fontdict=title_font)
            plt.scatter(diffs, diff_base, s=0.5**2)
            # plt.scatter(diffs, perf_list, s=0.5**2)
            plt.axhline(1, color="red", linestyle="--", linewidth=0.5)
            plt.tick_params(axis='x', labelsize=5)
            plt.tick_params(axis='y', labelsize=5)
            # if i != 8:
            #     plt.ylim([0, 100])
        
        plt.subplot(math.ceil(len(metrics_name) / 3), 3, len(metrics_name))
        plt.title("Histogram", fontdict=title_font)
        plt.hist(perf_list, bins=100, linewidth=0.5, color="green")
        plt.tick_params(axis='x', labelsize=5)
        plt.tick_params(axis='y', labelsize=5)
        plt.tight_layout()
        # plt.savefig(f"figure/figure_2-{name}.png", dpi=500)
        plt.savefig(f"figure/figure_2.png", dpi=500)
        plt.close()
#-----------------------------------------------------------------------------------------------
def get_metrics_cond(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):
    paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)
    
    metrics1 = [[] for _ in range(9)]
    metrics2 = [[] for _ in range(9)]
    for bi, ci, bfuse_exec in analysis:
        # Sort in ascending order
        for idx in range(9):
            if kernel1_metrics[bi][idx].value() > kernel2_metrics[ci][idx].value():
                metrics1[idx].append(kernel2_metrics[ci][idx].value())
                metrics2[idx].append(kernel1_metrics[bi][idx].value())
            else:
                metrics1[idx].append(kernel1_metrics[bi][idx].value())
                metrics2[idx].append(kernel2_metrics[ci][idx].value())
        # if kernel1_metrics[bi][1].value() > kernel2_metrics[ci][1].value(): # LG throttle
        #     for idx in range(9):
        #         metrics1[idx].append(kernel2_metrics[ci][idx].value())
        #         metrics2[idx].append(kernel1_metrics[bi][idx].value())
        # else:
        #     for idx in range(9):
        #         metrics1[idx].append(kernel1_metrics[bi][idx].value())
        #         metrics2[idx].append(kernel2_metrics[ci][idx].value())

    return metrics1, metrics2
#-----------------------------------------------------------------------------------------------
def draw_plot(infoYAML, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):

    # low
    metrics1_low, metrics2_low = get_metrics_cond(infoYAML, "low", lambda x: x < 0.8, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)
    # middle
    metrics1_middle, metrics2_middle = get_metrics_cond(infoYAML, "middle", lambda x: x >= 0.8 and x < 1.2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)
    # high
    metrics1_high, metrics2_high = get_metrics_cond(infoYAML, "high", lambda x: x >= 1.2, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    metrics_name = ["math_throttle", "lg_throttle", "mio_throttle", "tex_throttle",
                    "math_depend", "lg_depend", "mio_depend", "l1_tex_cache_hit", "bank_conflict"]
    
    # for idx in range(9):
    #     print(metrics_name[idx])
    #     for i in range(len(metrics1[idx])):
    #         print(f"{metrics1[idx][i]:.2f}", end=" ")
    #     print("")
    #     for i in range(len(metrics2[idx])):
    #         print(f"{metrics2[idx][i]:.2f}", end=" ")
    #     print("")
    #     print("")

    # plt.style.use('default')
    # plt.rcParams['figure.figsize'] = (4, 3)
    # plt.rcParams['font.size']      = 12

    for idx in range(9):
        ax1 = plt.subplot(1, 3, 1)
        plt.boxplot([metrics1_low[idx], metrics2_low[idx]], showfliers=False)
        plt.xlabel("low")

        ax2 = plt.subplot(1, 3, 2, sharey=ax1)
        plt.boxplot([metrics1_middle[idx], metrics2_middle[idx]], showfliers=False)
        plt.xlabel("middle")

        ax3 = plt.subplot(1, 3, 3, sharey=ax1)
        plt.boxplot([metrics1_high[idx], metrics2_high[idx]], showfliers=False)
        plt.xlabel("high")

        plt.tight_layout()
        plt.savefig(f"figure/plot-{metrics_name[idx]}.png", dpi=500)
        plt.close()
#-----------------------------------------------------------------------------------------------
def draw_heatmap_cond(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):
    paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

    # Draw stall heatmap
    print("Draw heatmap...")
    metrics_name = ["MATH_throttle", "LG_throttle", "MIO_throttle", "TEX_throttle"]
    for idx in range(4):
        bi_max_value = 0.
        ci_max_value = 0.
        bi_min_value = float("inf")
        ci_min_value = float("inf")

        for bi, ci, _ in analysis:
            if kernel1_metrics[bi][idx].value() > bi_max_value:
                bi_max_value = kernel1_metrics[bi][idx].value()
            if kernel1_metrics[bi][idx].value() < bi_min_value:
                bi_min_value = kernel1_metrics[bi][idx].value()

            if kernel2_metrics[ci][idx].value() > ci_max_value:
                ci_max_value = kernel2_metrics[ci][idx].value()
            if kernel2_metrics[ci][idx].value() < ci_min_value:
                ci_min_value = kernel2_metrics[ci][idx].value()

        bi_unit = (bi_max_value - bi_min_value) / len(kernel1_metrics)
        ci_unit = (ci_max_value - ci_min_value) / len(kernel2_metrics)

        tmp_data = []
        for i0 in range(len(kernel1_metrics)):
            tmp = []
            for i1 in range(len(kernel2_metrics)):
                tmp.append([])
            tmp_data.append(tmp)
        
        for bi, ci, _ in analysis:
            bi_value    = kernel1_metrics[bi][idx].value()
            ci_value    = kernel2_metrics[ci][idx].value()
            bfuse_value = bfuse_metrics[bi][ci][idx].value()

            if bi_value + ci_value == 0:
                continue

            # if bi_value == 0 and ci_value == 0:
            #     continue
            # elif bi_value == 0:
            #     diff = bfuse_metrics[bi][ci][idx].value() / kernel2_metrics[ci][idx].value()
            # elif ci_value == 0:
            #     diff = bfuse_metrics[bi][ci][idx].value() / kernel1_metrics[bi][idx].value()
            # else:
            #     diff1 = bfuse_metrics[bi][ci][idx].value() / kernel1_metrics[bi][idx].value()
            #     diff2 = bfuse_metrics[bi][ci][idx].value() / kernel2_metrics[ci][idx].value()
            #     diff = gmean([diff1, diff2])
            diff = bfuse_value
            # diff = bfuse_value / ((bi_value + ci_value) / 2)

            bi_idx = int((bi_value - bi_min_value) / bi_unit)
            ci_idx = int((ci_value - ci_min_value) / ci_unit)

            if bi_value == bi_max_value:
                bi_idx -= 1
            if ci_value == ci_max_value:
                ci_idx -= 1
            tmp_data[bi_idx][ci_idx].append(diff)

        data = np.empty((len(kernel1_metrics), len(kernel2_metrics)))
        for i0 in range(len(kernel1_metrics)):
            for i1 in range(len(kernel2_metrics)):
                if len(tmp_data[i0][i1]) == 0:
                    data[i0, i1] = 0.
                else:
                    data[i0, i1] = round(np.mean(tmp_data[i0][i1]), 2)

        bi_index = []
        ci_index = []
        for i in range(len(kernel1_metrics)):
            bi_index.append(f"{bi_min_value + bi_unit * i:.2f}")
        for i in range(len(kernel2_metrics)):
            ci_index.append(f"{ci_min_value + ci_unit * i:.2f}")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest')
        fig.colorbar(cax)

        ax.set_xticklabels(ci_index)
        ax.set_yticklabels(bi_index)

        plt.savefig(f"figure/heatmap-{name}-{metrics_name[idx]}.png", dpi=500)
        plt.close()
#-----------------------------------------------------------------------------------------------
def analyze_report(infoYAML):
    # Preprocess datas
    kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas = preprocess_datas(infoYAML)
    metrics_list, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics = preprocess_metrics(infoYAML)

    # # Case low-2
    # analyze_cond(infoYAML, "low-2", lambda x: x < 0.6, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Case low-1
    # analyze_cond(infoYAML, "low-1", lambda x: x >= 0.6 and x < 0.9, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Case middle
    # analyze_cond(infoYAML, "middle", lambda x: x >= 0.9 and x < 1.1, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Case high-1
    # analyze_cond(infoYAML, "high-1", lambda x: x >= 1.1 and x < 1.4, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Case high-2
    # analyze_cond(infoYAML, "high-2", lambda x: x >= 1.4, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Case all
    # analyze_cond(infoYAML, "all", lambda x: True, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # Find pattern
    pattern_cond2(infoYAML, "all", lambda x: True, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Draw boxplot all
    # draw_plot(infoYAML, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # # Draw heatmap all
    # draw_heatmap_cond(infoYAML, "all", lambda x: True, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)
#-----------------------------------------------------------------------------------------------
def print_metrics(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(1)

    kernel1_size   = len(fusion_sets[0]["Set"])
    kernel2_size   = len(fusion_sets[1]["Set"])

    report_path = os.path.join("metrics", f"0_0_0.ncu-rep")
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

    # Print metrics
    print_metrics(yaml_info)

    # Analyze ncu report results
    # analyze_report(yaml_info)
#-----------------------------------------------------------------------------------------------