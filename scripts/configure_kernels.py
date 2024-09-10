#!/usr/bin/python3

import yaml, json
import os, sys
import argparse
import numpy as np
import logging

import tvm
from tvm import relax
from tvm import dlight as dl

#-----------------------------------------------------------------------------------------------
def get_bgemm_shape(batch_size, M, K, N):
    bgemm_A_shape      = (batch_size, M, K)
    bgemm_B_shape      = (batch_size, N, K)
    bgemm_output_shape = (batch_size, M, N)
    
    return bgemm_A_shape, bgemm_B_shape, bgemm_output_shape
#-----------------------------------------------------------------------------------------------
def get_dense_shape(batch_size, in_dim, out_dim):
    dense_data_shape   = (batch_size, in_dim)
    dense_weight_shape = (out_dim, in_dim)
    dense_output_shape = (batch_size, out_dim)
    
    return dense_data_shape, dense_weight_shape, dense_output_shape
#-----------------------------------------------------------------------------------------------
def get_conv2d_shape(N, H, W, CO, CI, KH, KW, stride, padding):
    conv2d_data_shape   = (N, CI, H, W)
    conv2d_kernel_shape = (CO, CI, KH, KW)
    conv2d_output_shape = (N, CO,
                           int((H - KH + 2 * padding) / stride) + 1,
                           int((W - KW + 2 * padding) / stride) + 1)
    # conv2d_output_shape = (N, CO,
    #                        int((H - KH + 2 * padding[0]) / strides[0]) + 1,
    #                        int((W - KW + 2 * padding[1]) / strides[1]) + 1)
    
    return conv2d_data_shape, conv2d_kernel_shape, conv2d_output_shape
#-----------------------------------------------------------------------------------------------
def batch_matmul_module(batch_size, M, K, N, dtype="float32"):
    a = relax.Var("a", relax.TensorStructInfo([batch_size, M, K], dtype=dtype))
    b = relax.Var("b", relax.TensorStructInfo([batch_size, N, K], dtype=dtype))

    BB = relax.BlockBuilder()
    with BB.function("default_kernel", [a, b]):
        gv = BB.emit_te(tvm.topi.nn.batch_matmul, a, b, out_dtype=dtype)
        BB.emit_func_output(gv)
    return BB.get()
#-----------------------------------------------------------------------------------------------
def dense_module(batch_size, in_dim, out_dim, dtype="float32"):
    data   = relax.Var("x", relax.TensorStructInfo([batch_size, in_dim], dtype=dtype))
    weight = relax.Var("x", relax.TensorStructInfo([out_dim, in_dim], dtype=dtype))

    BB = relax.BlockBuilder()
    with BB.function("default_kernel", [data, weight]):
        gv = BB.emit_te(tvm.topi.nn.dense, data, weight, out_dtype=dtype)
        BB.emit_func_output(gv)
    return BB.get()
#-----------------------------------------------------------------------------------------------
def conv2d_module(N, H, W, CO, CI, KH, KW, strides, padding, dtype="float32"):
    x      = relax.Var("x", relax.TensorStructInfo([N, CI, H, W], dtype=dtype))
    weight = relax.Var("weight", relax.TensorStructInfo([CO, CI, KH, KW], dtype=dtype))

    BB = relax.BlockBuilder()
    with BB.function("default_kernel", [x, weight]):
        gv = BB.emit_te(tvm.topi.nn.conv2d, x, weight, dilation=1, strides=strides, padding=padding)
        BB.emit_func_output(gv)
    return BB.get()
#-----------------------------------------------------------------------------------------------
# Temporal variables
threads_info = [1, 1, 1, 1, 1, 1]
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
def compile_module(mod, target, callback_list):
    # Apply DLight rules
    with target:
        mod = tvm.ir.transform.Sequential(
            [
                relax.get_pipeline("zero"),
                # dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                #     dl.gpu.Matmul(),
                #     dl.gpu.GEMV(),
                #     dl.gpu.Reduction(),
                #     dl.gpu.GeneralReduction(),
                #     dl.gpu.Fallback(),
                # ),
            ]
        )(mod)

    builder  = relax.ExecBuilder()
    pipeline = relax.get_pipeline("default_build")

    with target:
        mod = pipeline(mod)
    mod = relax.vm_build._vmcodegen(builder, mod, "bytecode")

    tir_mod = relax.vm_build._filter_tir(mod)
    with tvm.transform.PassContext(opt_level=3, config={"tir.add_lower_pass": callback_list}):
        lib = tvm.build(tir_mod, target=target, runtime=relax.vm_build._autodetect_system_lib_req(target, system_lib=None))

    return lib
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
    # evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    evaluator = func.time_evaluator(func.entry_name, dev, number=1, repeat=1)

    return np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000
#-----------------------------------------------------------------------------------------------
def get_fusion_info(fusion_sets):
    kernel_list = fusion_sets[0]["Set"] + fusion_sets[1]["Set"]
    fusion_info = []

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        logging.error("Number of fusion sets are only 2.")
        exit(1)
    
    for s0 in fusion_sets[0]["Set"]:
        for s1 in fusion_sets[1]["Set"]:
            fusion_info.append([s0, s1])

    return kernel_list, fusion_info
#-----------------------------------------------------------------------------------------------
def get_kernel_info(kernel_list, infos, test_suite_path, eval):
    global threads_info

    kernel_info = {}
    cuda_code   = ""

    # Schedule & tuning kernel if needed
    for kname in kernel_list:
        kargs    = infos[kname]
        log_file = os.path.join(test_suite_path, "log", f"{kname}.json")

        if kname.startswith("bgemm"):
            shape = get_bgemm_shape(*kargs)
            mod   = batch_matmul_module(*kargs)
        elif kname.startswith("conv2d"):
            shape = get_conv2d_shape(*kargs)
            mod   = conv2d_module(*kargs)
        elif kname.startswith("dense"):
            shape = get_dense_shape(*kargs)
            mod   = dense_module(*kargs)
        else:
            logging.ERROR("Function and schedule with given kernel's name do not exist.")
            exit(1)

        # Compile tvm IRModule
        target = tvm.target.Target("cuda")
        lib = compile_module(mod, target=target, callback_list=[(1, extract_threads_info_callback)])

        # Get CUDA code
        code = str(lib.imported_modules[0].get_source())
        idx  = code.rfind("extern \"C\" __global__")
        if kname.startswith("bgemm"):
            code = code[idx:].replace("batch_matmul_kernel", kname)
        elif kname.startswith("conv2d"):
            code = code[idx:].replace("conv2d_kernel", kname)
        elif kname.startswith("dense"):
            code = code[idx:].replace("dense_kernel", kname)

        cuda_code = cuda_code + code

        # Store kernel's thread infomation
        kernel_info[kname] = threads_info
    
        # Evaluate kernel if needed
        if eval:
            print("Execution time of %s operator: %.3f ms" % (kname, evaluation(lib, shape)))

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
def get_test_suite(path, output, eval=True):
    # Settings
    # test_suite_name = ".".join(path.split("/")[-1].split(".")[:-1])
    test_suite_name        = output
    test_suite_path        = os.path.join(os.getcwd(), test_suite_name)
    test_suite_cuda_path   = os.path.join(test_suite_path, "cuda")
    test_suite_config_path = os.path.join(test_suite_path, "config")

    # Generate test suite directory
    if os.path.exists(test_suite_path):
        logging.error("\"%s\" alreay exists." % test_suite_path)
        exit(1)

    os.mkdir(test_suite_path)
    os.mkdir(test_suite_cuda_path)
    os.mkdir(test_suite_config_path)

    # Parse YAML files
    with open(path) as f:
        yaml_info = yaml.safe_load(f)

    sets  = yaml_info["FusionSet"]
    infos = yaml_info["KernelInfo"]

    # Get fusion info & kernel list
    kernel_list, fusion_info = get_fusion_info(sets)

    # Get kernel info & CUDA kernel code
    kernel_info, cuda_code = get_kernel_info(kernel_list, infos, test_suite_path, eval)

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
    parser.add_argument("-o", action="store", default="test-suite", dest="file",
                        help="output file name of generated test-suite")

    # Get arguments
    args   = parser.parse_args()
    config = args.config
    output = args.file
    eval   = args.eval
    
    if not os.path.exists(config):
        logging.error("Given config path \"%s\" doesn't exist." % config)
        exit(1)

    get_test_suite(config, output, eval)
#-----------------------------------------------------------------------------------------------