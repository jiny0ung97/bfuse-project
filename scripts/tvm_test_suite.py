#!/usr/bin/python3

import tvm
from tvm import auto_scheduler

import yaml
import os, argparse, shutil
import numpy as np
import logging

import tvm_schedules
#-----------------------------------------------------------------------------------------------
# Tuning trials
trials = 900

# Temporal variables
threads_info = [1, 1, 1, 1, 1, 1]

# Colors
color_red   = "\033[31m"
color_white = "\033[37m"
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
    task.tune(tune_option)

    return task
#-----------------------------------------------------------------------------------------------
def eval(func, data_shapes):
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
def get_test_suite(path, output, eval=True, tuning=False):
    global threads_info

    # Parse YAML files
    with open(path) as f:
        kernels = yaml.safe_load(f)

    # Settings
    # test_suite_name        = ".".join(path.split("/")[-1].split(".")[:-1])
    test_suite_name        = output
    test_suite_path        = os.path.join(os.getcwd(), test_suite_name)
    test_suite_thread_info = {}

    # Generate test suite directory
    if os.path.exists(test_suite_path):
        logging.warning("\"%s\" alreay exists." % test_suite_path)
        logging.warning("Remove \"%s\"." % test_suite_path)
        shutil.rmtree(test_suite_path)
    os.mkdir(test_suite_path)

    # Schedule & tuning kernel if needed
    for kernel in kernels:
        kname = kernel["KernelName"]
        kargs = kernel["Arguments"]
        log   = "%s/%s.json" % (test_suite_path, kname)

        if kname.startswith("bgemm"):
            shape = tvm_schedules.get_bgemm_shape(*kargs)
            if tuning:
                task      = auto_tuning(tvm_schedules.bgemm_workload, kargs, log)
                sch, args = task.apply_best(log)
            else:
                sch, args = tvm_schedules.cuda_schedule_bgemm(*kargs)
        elif kname.startswith("conv2d"):
            shape = tvm_schedules.get_conv2d_shape(*kargs)
            if tuning:
                task      = auto_tuning(tvm_schedules.conv2d_workload, kargs, log)
                sch, args = task.apply_best(log)
            else:
                sch, args = tvm_schedules.cuda_schedule_conv2d(*kargs)
        else:
            logging.ERROR("Function and schedule with given kernel's name do not exist.")
            exit(1)

        # Build & analysis kernel
        target = tvm.target.Target("cuda")
        if tuning:
            with auto_scheduler.ApplyHistoryBest(log):
                with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True,
                                                                    "tir.add_lower_pass": [(1, extract_threads_info_callback)]}):
                    func = tvm.build(sch, args, target)
        else:
            with tvm.transform.PassContext(opt_level=3, config={"tir.add_lower_pass": [(1, extract_threads_info_callback)]}):
                func = tvm.build(sch, args, target)

        # Store kernel's thread infomation
        test_suite_thread_info[kname] = threads_info
    
        # Evaluate kernel if needed
        if eval:
            print("Execution time of %s operator: %.3f ms" % (kname, eval(func, shape)))
    
    # Create kernel info YAML file
    yaml_dict = {}
    for kernel in kernels:
        kname  = kernel["KernelName"]
        T_info = test_suite_thread_info[kname]

        gridDim_dict = {
            "X": int(T_info[0]),
            "Y": int(T_info[1]),
            "Z": int(T_info[2]),
        }
        blockDim_dict = {
            "X": int(T_info[3]),
            "Y": int(T_info[4]),
            "Z": int(T_info[5]),
        }
        k_dict = {
            "KernelName": kname,
            "HasBarriers": True, # Don't care
            "GridDim": gridDim_dict,
            "BlockDim": blockDim_dict,
            "Reg": 32, # Don't care
            "ExecTime": -1, # Don't care
        }
        yaml_dict[kname] = k_dict
    
    config_yaml = os.path.join(test_suite_path, "kernels.yaml")
    with open(config_yaml, "w+") as f:
        yaml.dump(yaml_dict, f)
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", action="store",
                        help="config file path for generate test suite")
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