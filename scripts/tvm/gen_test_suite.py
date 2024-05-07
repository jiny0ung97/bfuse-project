#!/usr/bin/python3

import tvm
from tvm import relay, te, topi
from tvm import autotvm, auto_scheduler

import yaml
import os, sys, argparse
import numpy as np
import logging

from kernels import tvm_kernels
#-----------------------------------------------------------------------------------------------
# Tuning trials
trials = 900

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
def eval_default(sch, args, data_shapes, name):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    A_shape      = data_shapes[0]
    B_shape      = data_shapes[1]
    output_shape = data_shapes[2]

    # Build func & tensors
    func = tvm.build(sch, args, target)

    a_np = np.random.uniform(size=A_shape).astype(np.float32)
    b_np = np.random.uniform(size=B_shape).astype(np.float32)

    dev   = tvm.device(str(target))
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of %s operator: %.3f ms"
        % (name, np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
    )
#-----------------------------------------------------------------------------------------------
def eval_auto_tuning(sch, args, data_shapes, name, log):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    A_shape      = data_shapes[0]
    B_shape      = data_shapes[1]
    output_shape = data_shapes[2]

    # Build func & tensors
    with auto_scheduler.ApplyHistoryBest(log):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            func = tvm.build(sch, args, target)

    a_np = np.random.uniform(size=A_shape).astype(np.float32)
    b_np = np.random.uniform(size=B_shape).astype(np.float32)

    dev   = tvm.device(str(target))
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of %s operator: %.3f ms"
        % (name, np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
    )
#-----------------------------------------------------------------------------------------------
def get_test_suite(path, eval=True, tuning=False):
    # Parse YAML files
    with open(path) as f:
        kernels = yaml.safe_load(f)

    # Settings
    test_suite_name = ".".join(path.split("/")[-1].split(".")[:-1])
    test_suite_path = test_suite_name
    test_suite_log  = "%s.json" % test_suite_name

    # Generate test suite directory
    if os.path.exists(test_suite_path):
        logging.error("\"%s\" alreay exists." % test_suite_path)
        exit(1)
    os.mkdir(test_suite_path)

    # Extract kernels
    for kernel in kernels:
        kname = kernel["KernelName"]
        kargs = kernel["Arguments"]
        log   = ""

        if kname.startswith("bgemm"):
            shape = tvm_kernels.get_bgemm_shape(*kargs)
            if tuning:
                task = auto_tuning(tvm_kernels.bgemm_workload, kargs, log)
                sch, args = task.apply_best(log)
            else:
                sch, args = tvm_kernels.cuda_schedule_bgemm(*kargs)
        elif kname.startswith("conv2d"):
            shape = tvm_kernels.get_conv2d_shape(*kargs)
            if tuning:
                task = auto_tuning(tvm_kernels.conv2d_workload, kargs, log)
                sch, args = task.apply_best(log)
            else:
                sch, args = tvm_kernels.cuda_schedule_conv2d(*kargs)
        else:
            logging.ERROR("Function and schedule with given kernel's name do not exist.")
            exit(1)
        
        if eval:
            if tuning:
                eval_auto_tuning(sch, args, shape, kname, log)
            else:
                eval_default(sch, args, shape, kname)
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

    # Get arguments
    args   = parser.parse_args()
    config = args.config
    eval   = args.eval
    tuning = args.tuning
    
    if not os.path.exists(config):
        logging.error("Given config path \"%s\" doesn't exist." % config)
        exit(1)

    get_test_suite(config, eval=eval, tuning=tuning)
#-----------------------------------------------------------------------------------------------