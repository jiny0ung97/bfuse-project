#!/usr/bin/python3

import os
import logging
import yaml, csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
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

    print(f"================================================== {name.upper()} CASES ==================================================")
    print(f"Total num: {len(analysis)}")
    for idx, [bi, ci, bfuse_exec] in enumerate(analysis):
        if idx >= 10:
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
        kernel1_bank_conflict = f"{kernel1_metrics[bi][8].value()}"

        # Kernel2
        kernel2_math_throttle = f"{kernel2_metrics[ci][0].value():.2f}"
        kernel2_lg_throttle   = f"{kernel2_metrics[ci][1].value():.2f}"
        kernel2_mio_throttle  = f"{kernel2_metrics[ci][2].value():.2f}"
        kernel2_tex_throttle  = f"{kernel2_metrics[ci][3].value():.2f}"
        kernel2_math_depend   = f"{kernel2_metrics[ci][4].value():.2f}"
        kernel2_lg_depend     = f"{kernel2_metrics[ci][5].value():.2f}"
        kernel2_mio_depend    = f"{kernel2_metrics[ci][6].value():.2f}"
        kernel2_L1Tex_hit     = f"{kernel2_metrics[ci][7].value():.2f}"
        kernel2_bank_conflict = f"{kernel2_metrics[ci][8].value()}"

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
            hfuse_bank_conflict = f"{hfuse_metrics[bi][ci][8].value()}"
        else:
            hfuse_math_throttle = f"{hfuse_metrics[ci][bi][0].value():.2f}"
            hfuse_lg_throttle   = f"{hfuse_metrics[ci][bi][1].value():.2f}"
            hfuse_mio_throttle  = f"{hfuse_metrics[ci][bi][2].value():.2f}"
            hfuse_tex_throttle  = f"{hfuse_metrics[ci][bi][3].value():.2f}"
            hfuse_math_depend   = f"{hfuse_metrics[ci][bi][4].value():.2f}"
            hfuse_lg_depend     = f"{hfuse_metrics[ci][bi][5].value():.2f}"
            hfuse_mio_depend    = f"{hfuse_metrics[ci][bi][6].value():.2f}"
            hfuse_L1Tex_hit     = f"{hfuse_metrics[ci][bi][7].value():.2f}"
            hfuse_bank_conflict = f"{hfuse_metrics[ci][bi][8].value()}"

        # BFuse
        bfuse_math_throttle = f"{bfuse_metrics[bi][ci][0].value():.2f}"
        bfuse_lg_throttle   = f"{bfuse_metrics[bi][ci][1].value():.2f}"
        bfuse_mio_throttle  = f"{bfuse_metrics[bi][ci][2].value():.2f}"
        bfuse_tex_throttle  = f"{bfuse_metrics[bi][ci][3].value():.2f}"
        bfuse_math_depend   = f"{bfuse_metrics[bi][ci][4].value():.2f}"
        bfuse_lg_depend     = f"{bfuse_metrics[bi][ci][5].value():.2f}"
        bfuse_mio_depend    = f"{bfuse_metrics[bi][ci][6].value():.2f}"
        bfuse_L1Tex_hit     = f"{bfuse_metrics[bi][ci][7].value():.2f}"
        bfuse_bank_conflict = f"{bfuse_metrics[bi][ci][8].value()}"

        # Diff
        diff_metrics = []
        for idx in range(9):
            bi_value    = kernel1_metrics[bi][idx].value()
            ci_value    = kernel2_metrics[ci][idx].value()
            bfuse_value = bfuse_metrics[bi][ci][idx].value()

            if idx == 7:
                if bi_value == 0 and ci_value == 0:
                    diff = "NaN"
                elif bi_value == 0:
                    diff2 = bfuse_value / ci_value
                    diff  = f"{diff2:.2f}"
                elif ci_value == 0:
                    diff1 = bfuse_value / bi_value
                    diff  = f"{diff1:.2f}"
                else:
                    diff1 = bfuse_value / bi_value
                    diff2 = bfuse_value / ci_value
                    diff = f"{gmean([diff1, diff2]):.2f}"
            elif idx == 8:
                if bi_value + ci_value == 0:
                    diff = "NaN"
                else:
                    diff = f"{(bfuse_metrics[bi][ci][idx].value() / (kernel1_metrics[bi][idx].value() + kernel2_metrics[ci][idx].value())):.2f}"
            else:
                if bi_value + ci_value == 0:
                    diff = "NaN"
                else:
                    diff = f"{(bfuse_metrics[bi][ci][idx].value() / ((kernel1_metrics[bi][idx].value() + kernel2_metrics[ci][idx].value()) / 2)):.2f}"
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
#-----------------------------------------------------------------------------------------------
def draw_cond(infoYAML, name, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics):
    paralel, hfuse, bfuse, analysis = preprocess_by_cond(infoYAML, cond, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas)

    # Draw stall heatmap
    print("Draw heatmap...")
    metrics_name = ["MATH throttle", "LG throttle", "MIO throttle", "TEX throttle"]
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
            diff = bfuse_value / ((bi_value + ci_value) / 2)

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

        plt.savefig(f"heatmap-{metrics_name[idx]}.png", dpi=500)
#-----------------------------------------------------------------------------------------------
def analyze_report(infoYAML):
    # Preprocess datas
    kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas = preprocess_datas(infoYAML)
    metrics_list, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics = preprocess_metrics(infoYAML)

    # Case low
    analyze_cond(infoYAML, "low", lambda x: x < 0.8, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # Case middle
    analyze_cond(infoYAML, "middle", lambda x: x >= 0.9 and x < 1.1, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    # Case high
    analyze_cond(infoYAML, "high", lambda x: x >= 1.4, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)

    draw_cond(infoYAML, "all", lambda x: True, kernel1_datas, kernel2_datas, parallel_datas, hfuse_datas, bfuse_datas, kernel1_metrics, kernel2_metrics, hfuse_metrics, bfuse_metrics)
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
    # print_metrics(yaml_info)

    # Analyze ncu report results
    analyze_report(yaml_info)
#-----------------------------------------------------------------------------------------------