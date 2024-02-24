
import numpy as np
import tvm
from tvm import relay, te, topi
from tvm import autotvm, auto_scheduler

from kernel_schedules import schedule_conv2d
#--------------------------------------------------------------------------

#####################################
# Settings
#####################################

device_num = 3

conv2d_cu_file  = "cuda/conv2d.cu"
bgemm_cu_file   = "cuda/bgemm.cu"
softmax_cu_file = "cuda/softmax.cu"
pool2d_cu_file  = "cuda/pool2d.cu"

conv2d_tir_file  = "tir/conv2d.tir"
bgemm_tir_file   = "tir/bgemm.tir"
softmax_tir_file = "tir/softmax.cu"
pool2d_tir_file  = "tir/pool2d.cu"

conv2d_log_file  = "log/conv2d.json"

conv2d_python_file = "python/conv2d.py"

#####################################
# Kernel Arguments
#####################################

##### Resnet-18, Task 1 #####
# conv2d_kernel_args  = (128, 56, 56, 64, 64, 1, 1, 1, 1)      # (N, H, W, CO, CI, KH, KW, stride, padding) small
conv2d_kernel_args  = (64, 56, 56, 64, 64, 3, 3, 1, 1)      # (N, H, W, CO, CI, KH, KW, stride, padding) large
bgemm_kernel_args   = (32, 1, 512, 1000)                  # (batch_size, M, K, N)

softmax_kernel_args = (128, 1, 1000)                       # (batch_size, M, N)
pool2d_kernel_args  = (128, 56, 56, 64, 3, 3, 2, 0)        # (N, H, W, C, KH, KW, stride, padding)

#####################################
# Data Shape (NCHW)
#####################################

conv2d_data_shape   = (conv2d_kernel_args[0],
                       conv2d_kernel_args[4], # channel
                       conv2d_kernel_args[1], # height
                       conv2d_kernel_args[2]) # width

conv2d_kernel_shape = (conv2d_kernel_args[3], # out_channel
                       conv2d_kernel_args[4], # in_channel
                       conv2d_kernel_args[5], # height
                       conv2d_kernel_args[6]) # width

conv2d_output_shape = (conv2d_kernel_args[0],
                       conv2d_kernel_args[3],
                       (conv2d_kernel_args[1] - conv2d_kernel_args[5] + 2 * conv2d_kernel_args[8]) / conv2d_kernel_args[7] + 1,
                       (conv2d_kernel_args[2] - conv2d_kernel_args[6] + 2 * conv2d_kernel_args[8]) / conv2d_kernel_args[7] + 1)

bgemm_A_shape = (bgemm_kernel_args[0],
                 bgemm_kernel_args[1], # M
                 bgemm_kernel_args[2]) # K

bgemm_B_shape = (bgemm_kernel_args[0],
                 bgemm_kernel_args[3], # N
                 bgemm_kernel_args[2]) # K

bgemm_output_shape = (bgemm_kernel_args[0],
                      bgemm_kernel_args[1], # M
                      bgemm_kernel_args[3]) # N

softmax_data_shape = (softmax_kernel_args[0],
                      softmax_kernel_args[1], # M
                      softmax_kernel_args[2]) # N

softmax_output_shape = (softmax_kernel_args[0],
                        softmax_kernel_args[1], # M
                        softmax_kernel_args[2]) # N

pool2d_data_shape = (pool2d_kernel_args[0],
                     pool2d_kernel_args[3], # C
                     pool2d_kernel_args[1], # H
                     pool2d_kernel_args[2]) # W

pool2d_output_shape = (pool2d_kernel_args[0],
                       pool2d_kernel_args[3],
                       (pool2d_kernel_args[1] - pool2d_kernel_args[4] + 2 * pool2d_kernel_args[7]) /  pool2d_kernel_args[6] + 1,
                       (pool2d_kernel_args[2] - pool2d_kernel_args[5] + 2 * pool2d_kernel_args[7]) /  pool2d_kernel_args[6] + 1)

#--------------------------------------------------------------------------
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data   = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    out    = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    # with tvm.target.Target("cuda"):
    #     out = topi.cuda.conv2d_nchw(data, kernel, stride, padding, 1 ,"float32")
    #     sst = topi.cuda.schedule_conv2d_nchw([out])

    # return sst, [data, kernel, out]
    return [data, kernel, out]
#--------------------------------------------------------------------------
def conv2d_layer_tuning(log_file):
    # Target
    target = tvm.target.Target("cuda")

    # Extract search tasks
    print("Search tasks...")
    task = tvm.auto_scheduler.SearchTask(func=conv2d_layer,
                                         args=conv2d_kernel_args,
                                         target=target,
                                         )

    # Begin tuning
    print("Begin tuning...")
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    # task.tune(tune_option)

    return task
#--------------------------------------------------------------------------
def conv2d_layer_eval(sch, args):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    # data_shape   = conv2d_data_shape
    # kernel_shape = conv2d_kernel_shape
    # output_shape = conv2d_output_shape
    data_shape   = conv2d_data_shape[::-1]
    kernel_shape = conv2d_kernel_shape[::-1]
    output_shape = conv2d_output_shape[::-1]

    # Build func & tensors
    with auto_scheduler.ApplyHistoryBest(conv2d_log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            func = tvm.build(sch, args, target)

    a_np = np.random.uniform(size=data_shape).astype(np.float32)
    b_np = np.random.uniform(size=kernel_shape).astype(np.float32)

    dev   = tvm.device(str(target), device_num)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of conv2d operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def bgemm_layer(batch_size, M, K, N):
    A   = te.placeholder((batch_size, M, K), name="A")
    B   = te.placeholder((batch_size, N, K), name="B")
    # out = topi.nn.batch_matmul(A, B, transpose_b=True, out_dtype="float32")
    with tvm.target.Target("cuda"):
        out = topi.cuda.batch_matmul(A, B, (batch_size, M, N), "float32", False, True)
        sst = topi.cuda.schedule_batch_matmul([out])

    return sst, [A, B, out]
#--------------------------------------------------------------------------
def bgemm_layer_eval(sch, args):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    A_shape      = bgemm_A_shape
    B_shape      = bgemm_B_shape
    output_shape = bgemm_output_shape

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=A_shape).astype(np.float32)
    b_np = np.random.uniform(size=B_shape).astype(np.float32)

    dev   = tvm.device(str(target), device_num)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of bgemm operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def softmax_layer(batch_size, M, N):
    data = te.placeholder((batch_size, M, N), name="data")
    out  = topi.nn.softmax(data, axis=-1)
    with tvm.target.Target("cuda"):
        sst = topi.cuda.schedule_softmax(out)

    return sst, [data, out]
#--------------------------------------------------------------------------
def softmax_layer_eval(sch, args):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    data_shape   = softmax_data_shape
    output_shape = softmax_output_shape

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=data_shape).astype(np.float32)

    dev   = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of softmax operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def pool2d_layer(N, H, W, C, KH, KW, stride, padding):
    data = te.placeholder((N, C, H, W), name="data")
    out  = topi.nn.pool2d(data,
                          kernel=(KH, KW),
                          stride=(stride, stride),
                          dilation=(1, 1),
                          padding=(padding, padding, padding, padding),
                          pool_type="max",
                          layout="NCHW")
    with tvm.target.Target("cuda"):
        sst = topi.cuda.schedule_pool(out, "NCHW")

    return sst, [data, out]
#--------------------------------------------------------------------------
def pool2d_layer_eval(sch, args):
    # Target
    target = tvm.target.Target("cuda")

    # Parameters
    data_shape   = pool2d_data_shape
    output_shape = pool2d_output_shape

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=data_shape).astype(np.float32)

    dev   = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.empty(output_shape, device=dev)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of pool2d operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def get_cuda_code(sch, args, cu_file):
    # Target
    target = tvm.target.Target("cuda")

    # Get auto-tuned kernel code
    print("Get CUDA code...")
    with open(cu_file, "w+") as f:
        mod = tvm.build(sch, args, target=target)
        f.write(mod.imported_modules[0].get_source())    
#--------------------------------------------------------------------------
def get_tir_code(sch, args, tir_file):
    # Get TIR code
    print("Get Lowered TIR...")
    with open(tir_file, "w+") as f:
        f.write(str(tvm.lower(sch, args, simple_mode=True)))
#--------------------------------------------------------------------------
def get_python_code(task, log_file, python_file):
    # Get Python code
    print("Get Python code...")
    with open(python_file, "w+") as f:
        f.write(str(task.print_best(log_file)))
#--------------------------------------------------------------------------
        
        
################################################################################
# Evaluate layers (conv2d, bgemm)
# ----------------------------------

# test
sch, args = schedule_conv2d(*conv2d_kernel_args)
conv2d_layer_eval(sch, args)
get_tir_code(sch, args, conv2d_tir_file)
get_cuda_code(sch, args, conv2d_cu_file)

# # conv2d
# task = conv2d_layer_tuning(conv2d_log_file)
# sch, args = task.apply_best(conv2d_log_file)
# sch, args = conv2d_layer(*conv2d_kernel_args)
# conv2d_layer_eval(sch, args)
# get_tir_code(sch, args, conv2d_tir_file)
# get_cuda_code(sch, args, conv2d_cu_file)
# get_python_code(task, conv2d_log_file, conv2d_python_file)

# # bgemm
# sch, args = bgemm_layer(*bgemm_kernel_args)
# bgemm_layer_eval(sch, args)
# get_tir_code(sch, args, bgemm_tir_file)
# get_cuda_code(sch, args, bgemm_cu_file)

# # softmax
# sch, args = softmax_layer(*softmax_kernel_args)
# softmax_layer_eval(sch, args)
# get_tir_code(sch, args, softmax_tir_file)
# get_cuda_code(sch, args, softmax_cu_file)

# # pool2d
# sch, args = pool2d_layer(*pool2d_kernel_args)
# pool2d_layer_eval(sch, args)
# get_tir_code(sch, args, pool2d_tir_file)
# get_cuda_code(sch, args, pool2d_cu_file)