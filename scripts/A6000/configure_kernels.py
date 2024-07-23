#!/usr/bin/python3

import tvm
from tvm import auto_scheduler

import yaml, json
import os, sys
import argparse
import numpy as np
import logging

tvm_kernels_module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../test-utils/tvm-kernels")
sys.path.append(tvm_kernels_module_path)

import tvm_schedules

# #-------------------------- TEMP ----------------------------------
# from tvm import te
# from tvm import topi
# #-----------------------------------------------------------------------------------------------
# @auto_scheduler.register_workload
# def bgemm_workload(batch_size, M, K, N):
#     A   = te.placeholder((batch_size, M, K), name="A")
#     B   = te.placeholder((batch_size, N, K), name="B")

#     with tvm.target.Target("cuda"):
#         out = topi.cuda.batch_matmul(A, B, (batch_size, M, N), "float32", False, True)

#     return A, B, out
# #-----------------------------------------------------------------------------------------------
# @auto_scheduler.register_workload
# def conv2d_workload(N, H, W, CO, CI, KH, KW, stride, padding):
#     data   = te.placeholder((N, CI, H, W), name="data")
#     kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

#     with tvm.target.Target("cuda"):
#         out = topi.cuda.conv2d_nchw(data, kernel, stride, padding, 1 ,"float32")

#     return data, kernel, out
# #-------------------------- TEMP ----------------------------------

#-----------------------------------------------------------------------------------------------
# Tuning trials
trials = 900

# Temporal variables
threads_info = [1, 1, 1, 1, 1, 1]
#-----------------------------------------------------------------------------------------------
def auto_tuning(func, args, log_file):
    # Target
    target = tvm.target.Target("cuda")

    # Extract search tasks
    print("Search tasks...")
    task = tvm.auto_scheduler.SearchTask(func=func,
                                         args=args,
                                         target=target,
                                        )

    # Begin tuning
    print("Begin tuning...")
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=trials,
    #     runner=auto_scheduler.RPCRunner("A6000", "127.0.0.1", 9190),
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    #     verbose=2,
    # )
    # task.tune(tune_option)

    return task
#-----------------------------------------------------------------------------------------------
def evaluation(func, data_shapes):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    A_shape      = data_shapes[0]
    B_shape      = data_shapes[1]
    output_shape = data_shapes[2]

    a_np = np.random.uniform(size=A_shape).astype(np.float32)
    b_np = np.random.uniform(size=B_shape).astype(np.float32)

    dev   = tvm.device(str(target))
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)

    return np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000
#-----------------------------------------------------------------------------------------------
def extract_threads_info(op):
    global threads_info

    if isinstance(op, tvm.tir.AttrStmt):
        if op.attr_key == "thread_extent":
            tag = op.node.thread_tag
            if tag == "blockIdx.x":
                threads_info[0] = op.value
            elif tag == "blockIdx.y":
                threads_info[1] = op.value
            elif tag == "blockIdx.z":
                threads_info[2] = op.value
            elif tag == "threadIdx.x":
                threads_info[3] = op.value
            elif tag == "threadIdx.y":
                threads_info[4] = op.value
            elif tag == "threadIdx.z":
                threads_info[5] = op.value
#-----------------------------------------------------------------------------------------------
@tvm.tir.transform.prim_func_pass(opt_level=0)
def extract_threads_info_callback(f, mod, ctx):
    global threads_info

    # Initialize & visit
    threads_info = [1, 1, 1, 1, 1, 1]
    tvm.tir.stmt_functor.post_order_visit(f.body, extract_threads_info)
    return f
#-----------------------------------------------------------------------------------------------
def get_cuda_code(sch, args):
    # Target
    target = tvm.target.Target("cuda")

    # Get auto-tuned kernel code
    mod = tvm.build(sch, args, target=target)

    return str(mod.imported_modules[0].get_source())
#-----------------------------------------------------------------------------------------------
def get_fusion_info(fusion_sets):
    kernel_list = fusion_sets[0]["Set"] + fusion_sets[1]["Set"]
    fusion_info = []

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(1)
    
    for s0 in fusion_sets[0]["Set"]:
        for s1 in fusion_sets[1]["Set"]:
            fusion_info.append([s0, s1])

    return kernel_list, fusion_info
#-----------------------------------------------------------------------------------------------
def get_kernel_info(kernel_list, infos, test_suite_path, eval, tuning):
    global threads_info

    kernel_info = {}
    cuda_code   = ""

    # Schedule & tuning kernel if needed
    for kname in kernel_list:
        kargs    = infos[kname]
        log_file = os.path.join(test_suite_path, "log", f"{kname}.json")

        if kname.startswith("bgemm"):
            shape = tvm_schedules.get_bgemm_shape(*kargs)
            if tuning:
                task      = auto_tuning(tvm_schedules.bgemm_workload, kargs, log_file)
                # task      = auto_tuning(bgemm_workload, kargs, log_file)
                sch, args = task.apply_best(log_file)
            else:
                sch, args = tvm_schedules.cuda_schedule_bgemm(*kargs)
        elif kname.startswith("conv2d"):
            shape = tvm_schedules.get_conv2d_shape(*kargs)
            if tuning:
                task      = auto_tuning(tvm_schedules.conv2d_workload, kargs, log_file)
                # task      = auto_tuning(conv2d_workload, kargs, log_file)
                sch, args = task.apply_best(log_file)
            else:
                sch, args = tvm_schedules.cuda_schedule_conv2d(*kargs)
        else:
            logging.ERROR("Function and schedule with given kernel's name do not exist.")
            exit(1)

        # Build & analysis kernel
        target = tvm.target.Target("cuda")
        if tuning:
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True,
                                                                    "tir.add_lower_pass": [(1, extract_threads_info_callback)]}):
                    func = tvm.build(sch, args, target)
        else:
            with tvm.transform.PassContext(opt_level=3, config={"tir.add_lower_pass": [(1, extract_threads_info_callback)]}):
                func = tvm.build(sch, args, target)

        # Store kernel's thread infomation
        kernel_info[kname] = threads_info

        # Get CUDA code
        code = get_cuda_code(sch, args)
        idx  = code.rfind("extern \"C\" __global__")
        code = code[idx:]

        code      = code.replace("default_function_kernel", kname)
        cuda_code = cuda_code + code
    
        # Evaluate kernel if needed
        if eval:
            print("Execution time of %s operator: %.3f ms" % (kname, evaluation(func, shape)))

    return kernel_info, cuda_code
#-----------------------------------------------------------------------------------------------
def get_compile_commands(test_suite_cuda_path, file):
    compile_commands_json = []
    file_path             = os.path.join(test_suite_cuda_path, file)

    compile_commands = {
        "directory": test_suite_cuda_path,
        "file": file_path,
        "command": "clang++-16 " + file_path + " -resource-dir /usr/lib/clang/16 --cuda-gpu-arch=sm_70 --cuda-device-only -pthread"
    }
    compile_commands_json.append(compile_commands)

    return compile_commands_json
#-----------------------------------------------------------------------------------------------
def get_test_suite(path, output, eval=True, tuning=False):
    # Settings
    # test_suite_name = ".".join(path.split("/")[-1].split(".")[:-1])
    test_suite_name        = output
    test_suite_path        = os.path.join(os.getcwd(), test_suite_name)
    test_suite_cuda_path   = os.path.join(test_suite_path, "cuda")
    test_suite_config_path = os.path.join(test_suite_path, "config")
    test_suite_log_path    = os.path.join(test_suite_path, "log")

    # Generate test suite directory
    if os.path.exists(test_suite_path):
        logging.error("\"%s\" alreay exists." % test_suite_path)
        exit(1)

    os.mkdir(test_suite_path)
    os.mkdir(test_suite_cuda_path)
    os.mkdir(test_suite_config_path)
    if tuning:
        os.mkdir(test_suite_log_path)

    # Parse YAML files
    with open(path) as f:
        yaml_info = yaml.safe_load(f)

    sets  = yaml_info["FusionSet"]
    infos = yaml_info["KernelInfo"]

    # Get fusion info & kernel list
    kernel_list, fusion_info = get_fusion_info(sets)

    # Get kernel info & CUDA kernel code
    kernel_info, cuda_code = get_kernel_info(kernel_list, infos, test_suite_path, eval, tuning)

    # Get compile commands
    file = "kernels.cu"
    compile_commands_json = get_compile_commands(test_suite_cuda_path, file)

    # Create CUDA kernel file
    code_path = "%s/kernels.cu" % (test_suite_cuda_path)
    with open(code_path, "w+") as f:
        f.write(cuda_code)

    # Create compile_commands.json file
    compile_commands_path = "%s/compile_commands.json" % (test_suite_cuda_path)
    with open(compile_commands_path, "w+") as f:
        json.dump(compile_commands_json, f)

    # Create fusion info YAML file
    fusion_yaml_list = []
    fusion_code_path = code_path.split("/")[-1]
    for f_info in fusion_info:
        f_dict = {
            "File": fusion_code_path,
            "Kernels": f_info
        }
        fusion_yaml_list.append(f_dict)

    fusion_config_yaml = os.path.join(test_suite_config_path, "fusions.yaml")
    with open(fusion_config_yaml, "w+") as f:
        yaml.dump(fusion_yaml_list, f)
    
    # Create kernel info YAML file
    kernel_yaml_dict = {}
    for kname in kernel_list:
        t_info = kernel_info[kname]

        gridDim_dict = {
            "X": int(t_info[0]),
            "Y": int(t_info[1]),
            "Z": int(t_info[2]),
        }
        blockDim_dict = {
            "X": int(t_info[3]),
            "Y": int(t_info[4]),
            "Z": int(t_info[5]),
        }
        k_dict = {
            "KernelName": kname,
            "HasBarriers": True, # Don't care
            "GridDim": gridDim_dict,
            "BlockDim": blockDim_dict,
            "Reg": 32, # Don't care
            "ExecTime": -1, # Don't care
        }
        kernel_yaml_dict[kname] = k_dict
    
    kernel_config_yaml = os.path.join(test_suite_config_path, "kernels.yaml")
    with open(kernel_config_yaml, "w+") as f:
        yaml.dump(kernel_yaml_dict, f)

    # Copy input YAML file
    input_config_yaml = os.path.join(test_suite_config_path, "info.yaml")
    with open(input_config_yaml, "w+") as f:
        yaml.dump(yaml_info, f)
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", action="store", help="config file path for generate test suite")
    parser.add_argument("-e", action="store_true", default=False, dest="eval",
                        help="run evaluation with test suite kernels")
    parser.add_argument("-t", action="store_true", default=False, dest="tuning",
                        help="use auto-tuning to generate test suite (a.k.a Ansor)")
    parser.add_argument("-o", action="store", default="test-suite", dest="file",
                        help="output file name of generated test-suite")

    # Get arguments
    args   = parser.parse_args()
    config = args.config
    output = args.file
    eval   = args.eval
    tuning = args.tuning
    
    if not os.path.exists(config):
        logging.error("Given config path \"%s\" doesn't exist." % config)
        exit(1)

    get_test_suite(config, output, eval=eval, tuning=tuning)
#-----------------------------------------------------------------------------------------------