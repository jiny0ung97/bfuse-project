
import numpy as np
import tvm
from tvm import relay, te, topi
from tvm import auto_scheduler
#--------------------------------------------------------------------------
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data   = te.placeholder((N, H, W, CI), name="data")
    kernel = te.placeholder((KH, KW, CI, CO), name="kernel")
    out    = topi.nn.conv2d_nhwc(data, kernel, stride, padding, dilation=1, out_dtype="float32")

    return [data, kernel, out]
#--------------------------------------------------------------------------
@auto_scheduler.register_workload
def bgemm_layer(batch_size, M, K, N):
    A   = te.placeholder((batch_size, M, K), name="A")
    B   = te.placeholder((batch_size, N, K), name="B")
    out = topi.nn.batch_matmul(A, B, transpose_b=True, out_dtype="float32")

    return [A, B, out]
#--------------------------------------------------------------------------
@auto_scheduler.register_workload
def softmax_layer(batch_size, M, N):
    data = te.placeholder((batch_size, M, N), name="data")
    out  = topi.nn.softmax(data, axis=-1)

    return [data, out]
#--------------------------------------------------------------------------
@auto_scheduler.register_workload
def pool2d_layer(N, H, W, C, KH, KW, stride, padding):
    data = te.placeholder((N, H, W, C), name="data")
    out  = topi.nn.pool2d(data,
                          kernel=(KH, KW),
                          stride=(stride, stride),
                          dilation=(1, 1),
                          padding=(padding, padding, padding, padding),
                          pool_type="max",
                          layout="NHWC")

    return [data, out]
#--------------------------------------------------------------------------
def conv2d_layer_tuning(batch_size, target, log_file):
    # Extract search tasks
    print("Search tasks...")
    task = tvm.auto_scheduler.SearchTask(func=conv2d_layer,
                                         args=(batch_size, 56, 56, 128, 64, 3, 3, 2, 1),
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
def bgemm_layer_tuning(batch_size, target, log_file):
    # Extract search tasks
    print("Search tasks...")
    task = tvm.auto_scheduler.SearchTask(func=bgemm_layer,
                                         args=(batch_size, 1024, 1024, 1024),
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
def softmax_layer_tuning(batch_size, target, log_file):
    # Extract search tasks
    print("Search tasks...")
    task = tvm.auto_scheduler.SearchTask(func=softmax_layer,
                                         args=(batch_size, 1, 1000),
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
def pool2d_layer_tuning(batch_size, target, log_file):
    # Extract search tasks
    print("Search tasks...")
    task = tvm.auto_scheduler.SearchTask(func=pool2d_layer,
                                         args=(batch_size, 56, 56, 64, 3, 3, 2, 0),
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
def analyze_task(task, log_file):
    # Print compute graph
    print("Computational DAG:")
    print("========== Task (workload key: %s) ==========" % (task.workload_key))
    print(task.compute_dag)

    # Print python code of best scheduling
    print("Best python scheduling code:")
    print(task.print_best(log_file))
#--------------------------------------------------------------------------
def analyze_sch(sch, args, target, cu_file, cuda_save=False):
    # Print lowered tir code
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    # Get auto-tuned kernel code
    if cuda_save:
        print("Get Auto-tuned kernel code...")
        with open(cu_file, "w+") as f:
            mod = tvm.build(sch, args, target=target)
            f.write(mod.imported_modules[0].get_source())
#--------------------------------------------------------------------------
def conv2d_layer_eval(batch_size, sch, args, target):
    # Parameters
    data_shape   = (batch_size, 56, 56, 64)  # (batch, height, width, channel)
    kernel_shape = (3, 3, 64, 128)           # (height, width, in_channel, out_channel)
    output_shape = (batch_size, 28, 28, 128)

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=data_shape).astype(np.float32)
    b_np = np.random.uniform(size=kernel_shape).astype(np.float32)

    dev   = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)

    # Execute tvm function
    # func(a_tvm, b_tvm, c_tvm)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of conv2d operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def bgemm_layer_eval(batch_size, sch, args, target):
    # Parameters
    A_shape      = (batch_size, 1024, 1024) # (batch, M, K)
    B_shape      = (batch_size, 1024, 1024) # (batch, N, K)
    output_shape = (batch_size, 1024, 1024)

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=A_shape).astype(np.float32)
    b_np = np.random.uniform(size=B_shape).astype(np.float32)

    dev   = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.empty(output_shape, device=dev)
    
    # Execute tvm function
    # func(a_tvm, b_tvm, c_tvm)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of bgemm operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, c_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def softmax_layer_eval(batch_size, sch, args, target):
    # Parameters
    data_shape   = (batch_size, 1, 1000) # (batch, M, N)
    output_shape = (batch_size, 1, 1000)

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=data_shape).astype(np.float32)

    dev   = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.empty(output_shape, device=dev)

    # Execute tvm function
    # func(a_tvm, b_tvm)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of softmax operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def pool2d_layer_eval(batch_size, sch, args, target):
    # Parameters
    data_shape   = (batch_size, 56, 56, 64) # (batch, height, width, channel)
    output_shape = (batch_size, 27, 27, 64)

    # Build func & tensors
    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=data_shape).astype(np.float32)

    dev   = tvm.device(str(target), 0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.empty(output_shape, device=dev)

    # Execute tvm function
    # func(a_tvm, b_tvm)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=10)
    print(
        "Execution time of pool2d operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm).results) * 1000)
    )
#--------------------------------------------------------------------------
def conv2d_layer_schedule(batch_size, target, cu_save=False):
    args = conv2d_layer(batch_size, 56, 56, 128, 64, 3, 3, 2, 1)
    s    = te.create_schedule(args[2].op)

    # ========== Task (workload key: ["conv2d_layer", 128, 56, 56, 128, 64, 3, 3, 2, 1]) ==========
    # data = PLACEHOLDER [128, 56, 56, 64]
    # pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), data[i0, (i1 - 1), (i2 - 1), i3], 0f)
    # kernel = PLACEHOLDER [3, 3, 64, 128]
    # conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*kernel[ry, rx, rc, ff])

    # Parameters
    conv2d_nhwc = args[2]                                     # args[2]
    pad_temp    = conv2d_nhwc.op.body[0].source[0].a.producer
    kernel      = conv2d_nhwc.op.body[0].source[0].b.producer # args[1]
    data        = pad_temp.op.body[0].args[1].producer        # args[0]

    assert(args[2] == conv2d_nhwc and
           args[1] == kernel and
           args[0] == data)

    pad_temp_i0, pad_temp_i1, pad_temp_i2, pad_temp_i3 = tuple(pad_temp.op.axis) + tuple(pad_temp.op.reduce_axis)
    conv2d_nhwc_nn, conv2d_nhwc_yy, conv2d_nhwc_xx, conv2d_nhwc_ff, conv2d_nhwc_ry, conv2d_nhwc_rx, conv2d_nhwc_rc = tuple(conv2d_nhwc.op.axis) + tuple(conv2d_nhwc.op.reduce_axis)
    conv2d_nhwc_local, = s.cache_write([conv2d_nhwc], "local")
    conv2d_nhwc_local_nn_c, conv2d_nhwc_local_yy_c, conv2d_nhwc_local_xx_c, conv2d_nhwc_local_ff_c, conv2d_nhwc_local_ry, conv2d_nhwc_local_rx, conv2d_nhwc_local_rc = tuple(conv2d_nhwc_local.op.axis) + tuple(conv2d_nhwc_local.op.reduce_axis)
    conv2d_nhwc_local_nn_c_o_i, conv2d_nhwc_local_nn_c_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_nn_c, factor=1)
    conv2d_nhwc_local_nn_c_o_o_i, conv2d_nhwc_local_nn_c_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_nn_c_o_i, factor=4)
    conv2d_nhwc_local_nn_c_o_o_o_i, conv2d_nhwc_local_nn_c_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_nn_c_o_o_i, factor=1)
    conv2d_nhwc_local_nn_c_o_o_o_o, conv2d_nhwc_local_nn_c_o_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_nn_c_o_o_o_i, factor=1)
    conv2d_nhwc_local_yy_c_o_i, conv2d_nhwc_local_yy_c_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_yy_c, factor=1)
    conv2d_nhwc_local_yy_c_o_o_i, conv2d_nhwc_local_yy_c_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_yy_c_o_i, factor=1)
    conv2d_nhwc_local_yy_c_o_o_o_i, conv2d_nhwc_local_yy_c_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_yy_c_o_o_i, factor=4)
    conv2d_nhwc_local_yy_c_o_o_o_o, conv2d_nhwc_local_yy_c_o_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_yy_c_o_o_o_i, factor=1)
    conv2d_nhwc_local_xx_c_o_i, conv2d_nhwc_local_xx_c_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_xx_c, factor=2)
    conv2d_nhwc_local_xx_c_o_o_i, conv2d_nhwc_local_xx_c_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_xx_c_o_i, factor=2)
    conv2d_nhwc_local_xx_c_o_o_o_i, conv2d_nhwc_local_xx_c_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_xx_c_o_o_i, factor=1)
    conv2d_nhwc_local_xx_c_o_o_o_o, conv2d_nhwc_local_xx_c_o_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_xx_c_o_o_o_i, factor=1)
    conv2d_nhwc_local_ff_c_o_i, conv2d_nhwc_local_ff_c_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_ff_c, factor=1)
    conv2d_nhwc_local_ff_c_o_o_i, conv2d_nhwc_local_ff_c_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_ff_c_o_i, factor=2)
    conv2d_nhwc_local_ff_c_o_o_o_i, conv2d_nhwc_local_ff_c_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_ff_c_o_o_i, factor=64)
    conv2d_nhwc_local_ff_c_o_o_o_o, conv2d_nhwc_local_ff_c_o_o_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_ff_c_o_o_o_i, factor=1)
    conv2d_nhwc_local_ry_o_i, conv2d_nhwc_local_ry_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_ry, factor=1)
    conv2d_nhwc_local_ry_o_o, conv2d_nhwc_local_ry_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_ry_o_i, factor=3)
    conv2d_nhwc_local_rx_o_i, conv2d_nhwc_local_rx_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_rx, factor=3)
    conv2d_nhwc_local_rx_o_o, conv2d_nhwc_local_rx_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_rx_o_i, factor=1)
    conv2d_nhwc_local_rc_o_i, conv2d_nhwc_local_rc_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_rc, factor=4)
    conv2d_nhwc_local_rc_o_o, conv2d_nhwc_local_rc_o_i = s[conv2d_nhwc_local].split(conv2d_nhwc_local_rc_o_i, factor=2)
    s[conv2d_nhwc_local].reorder(conv2d_nhwc_local_nn_c_o_o_o_o, conv2d_nhwc_local_yy_c_o_o_o_o, conv2d_nhwc_local_xx_c_o_o_o_o, conv2d_nhwc_local_ff_c_o_o_o_o, conv2d_nhwc_local_nn_c_o_o_o_i, conv2d_nhwc_local_yy_c_o_o_o_i, conv2d_nhwc_local_xx_c_o_o_o_i, conv2d_nhwc_local_ff_c_o_o_o_i, conv2d_nhwc_local_nn_c_o_o_i, conv2d_nhwc_local_yy_c_o_o_i, conv2d_nhwc_local_xx_c_o_o_i, conv2d_nhwc_local_ff_c_o_o_i, conv2d_nhwc_local_ry_o_o, conv2d_nhwc_local_rx_o_o, conv2d_nhwc_local_rc_o_o, conv2d_nhwc_local_ry_o_i, conv2d_nhwc_local_rx_o_i, conv2d_nhwc_local_rc_o_i, conv2d_nhwc_local_nn_c_o_i, conv2d_nhwc_local_yy_c_o_i, conv2d_nhwc_local_xx_c_o_i, conv2d_nhwc_local_ff_c_o_i, conv2d_nhwc_local_ry_i, conv2d_nhwc_local_rx_i, conv2d_nhwc_local_rc_i, conv2d_nhwc_local_nn_c_i, conv2d_nhwc_local_yy_c_i, conv2d_nhwc_local_xx_c_i, conv2d_nhwc_local_ff_c_i)
    conv2d_nhwc_nn_o_i, conv2d_nhwc_nn_i = s[conv2d_nhwc].split(conv2d_nhwc_nn, factor=4)
    conv2d_nhwc_nn_o_o_i, conv2d_nhwc_nn_o_i = s[conv2d_nhwc].split(conv2d_nhwc_nn_o_i, factor=1)
    conv2d_nhwc_nn_o_o_o, conv2d_nhwc_nn_o_o_i = s[conv2d_nhwc].split(conv2d_nhwc_nn_o_o_i, factor=1)
    conv2d_nhwc_yy_o_i, conv2d_nhwc_yy_i = s[conv2d_nhwc].split(conv2d_nhwc_yy, factor=1)
    conv2d_nhwc_yy_o_o_i, conv2d_nhwc_yy_o_i = s[conv2d_nhwc].split(conv2d_nhwc_yy_o_i, factor=4)
    conv2d_nhwc_yy_o_o_o, conv2d_nhwc_yy_o_o_i = s[conv2d_nhwc].split(conv2d_nhwc_yy_o_o_i, factor=1)
    conv2d_nhwc_xx_o_i, conv2d_nhwc_xx_i = s[conv2d_nhwc].split(conv2d_nhwc_xx, factor=4)
    conv2d_nhwc_xx_o_o_i, conv2d_nhwc_xx_o_i = s[conv2d_nhwc].split(conv2d_nhwc_xx_o_i, factor=1)
    conv2d_nhwc_xx_o_o_o, conv2d_nhwc_xx_o_o_i = s[conv2d_nhwc].split(conv2d_nhwc_xx_o_o_i, factor=1)
    conv2d_nhwc_ff_o_i, conv2d_nhwc_ff_i = s[conv2d_nhwc].split(conv2d_nhwc_ff, factor=2)
    conv2d_nhwc_ff_o_o_i, conv2d_nhwc_ff_o_i = s[conv2d_nhwc].split(conv2d_nhwc_ff_o_i, factor=64)
    conv2d_nhwc_ff_o_o_o, conv2d_nhwc_ff_o_o_i = s[conv2d_nhwc].split(conv2d_nhwc_ff_o_o_i, factor=1)
    s[conv2d_nhwc].reorder(conv2d_nhwc_nn_o_o_o, conv2d_nhwc_yy_o_o_o, conv2d_nhwc_xx_o_o_o, conv2d_nhwc_ff_o_o_o, conv2d_nhwc_nn_o_o_i, conv2d_nhwc_yy_o_o_i, conv2d_nhwc_xx_o_o_i, conv2d_nhwc_ff_o_o_i, conv2d_nhwc_nn_o_i, conv2d_nhwc_yy_o_i, conv2d_nhwc_xx_o_i, conv2d_nhwc_ff_o_i, conv2d_nhwc_nn_i, conv2d_nhwc_yy_i, conv2d_nhwc_xx_i, conv2d_nhwc_ff_i)
    s[conv2d_nhwc_local].compute_at(s[conv2d_nhwc], conv2d_nhwc_ff_o_i)
    kernel_shared = s.cache_read(kernel, "shared", [conv2d_nhwc_local])
    kernel_shared_ax0, kernel_shared_ax1, kernel_shared_ax2, kernel_shared_ax3 = tuple(kernel_shared.op.axis)
    s[kernel_shared].compute_at(s[conv2d_nhwc_local], conv2d_nhwc_local_rc_o_o)
    pad_temp_shared = s.cache_read(pad_temp, "shared", [conv2d_nhwc_local])
    pad_temp_shared_ax0, pad_temp_shared_ax1, pad_temp_shared_ax2, pad_temp_shared_ax3 = tuple(pad_temp_shared.op.axis)
    s[pad_temp_shared].compute_at(s[conv2d_nhwc_local], conv2d_nhwc_local_rc_o_o)
    s[pad_temp].compute_inline()
    conv2d_nhwc_nn_o_o_o_yy_o_o_o_fused_xx_o_o_o_fused_ff_o_o_o_fused = s[conv2d_nhwc].fuse(conv2d_nhwc_nn_o_o_o, conv2d_nhwc_yy_o_o_o, conv2d_nhwc_xx_o_o_o, conv2d_nhwc_ff_o_o_o)
    s[conv2d_nhwc].bind(conv2d_nhwc_nn_o_o_o_yy_o_o_o_fused_xx_o_o_o_fused_ff_o_o_o_fused, te.thread_axis("blockIdx.x"))
    conv2d_nhwc_nn_o_o_i_yy_o_o_i_fused_xx_o_o_i_fused_ff_o_o_i_fused = s[conv2d_nhwc].fuse(conv2d_nhwc_nn_o_o_i, conv2d_nhwc_yy_o_o_i, conv2d_nhwc_xx_o_o_i, conv2d_nhwc_ff_o_o_i)
    s[conv2d_nhwc].bind(conv2d_nhwc_nn_o_o_i_yy_o_o_i_fused_xx_o_o_i_fused_ff_o_o_i_fused, te.thread_axis("vthread"))
    conv2d_nhwc_nn_o_i_yy_o_i_fused_xx_o_i_fused_ff_o_i_fused = s[conv2d_nhwc].fuse(conv2d_nhwc_nn_o_i, conv2d_nhwc_yy_o_i, conv2d_nhwc_xx_o_i, conv2d_nhwc_ff_o_i)
    s[conv2d_nhwc].bind(conv2d_nhwc_nn_o_i_yy_o_i_fused_xx_o_i_fused_ff_o_i_fused, te.thread_axis("threadIdx.x"))
    kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused = s[kernel_shared].fuse(kernel_shared_ax0, kernel_shared_ax1, kernel_shared_ax2, kernel_shared_ax3)
    kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i = s[kernel_shared].split(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused, factor=4)
    s[kernel_shared].vectorize(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i)
    kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_o, kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i = s[kernel_shared].split(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, factor=256)
    s[kernel_shared].bind(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i, te.thread_axis("threadIdx.x"))
    pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused = s[pad_temp_shared].fuse(pad_temp_shared_ax0, pad_temp_shared_ax1, pad_temp_shared_ax2, pad_temp_shared_ax3)
    pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i = s[pad_temp_shared].split(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused, factor=4)
    s[pad_temp_shared].vectorize(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i)
    pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_o, pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i = s[pad_temp_shared].split(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, factor=256)
    s[pad_temp_shared].bind(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i, te.thread_axis("threadIdx.x"))
    s[conv2d_nhwc_local].pragma(conv2d_nhwc_local_nn_c_o_o_o_o, "auto_unroll_max_step", 512)
    s[conv2d_nhwc_local].pragma(conv2d_nhwc_local_nn_c_o_o_o_o, "unroll_explicit", True)

    return s, args
#--------------------------------------------------------------------------
def bgemm_layer_schedule(batch_size, target, cu_save=False):
    args = bgemm_layer(batch_size, 1024, 1024, 1024)
    s    = te.create_schedule(args[2].op)

    # ========== Task (workload key: ["bgemm_layer", 128, 1024, 1024, 1024]) ==========
    # A = PLACEHOLDER [128, 1024, 1024]
    # B = PLACEHOLDER [128, 1024, 1024]
    # T_batch_matmul_NT(b, i, j) += (A[b, i, k]*B[b, j, k])

    # Parameters
    T_batch_matmul_NT = args[2]                           # args[2]
    B = T_batch_matmul_NT.op.body[0].source[0].b.producer # args[1]
    A = T_batch_matmul_NT.op.body[0].source[0].a.producer # args[0]

    assert(args[2] == T_batch_matmul_NT and
           args[1] == B and
           args[0] == A)
    
    T_batch_matmul_NT_b, T_batch_matmul_NT_i, T_batch_matmul_NT_j, T_batch_matmul_NT_k = tuple(T_batch_matmul_NT.op.axis) + tuple(T_batch_matmul_NT.op.reduce_axis)
    T_batch_matmul_NT_local, = s.cache_write([T_batch_matmul_NT], "local")
    T_batch_matmul_NT_local_b_c, T_batch_matmul_NT_local_i_c, T_batch_matmul_NT_local_j_c, T_batch_matmul_NT_local_k = tuple(T_batch_matmul_NT_local.op.axis) + tuple(T_batch_matmul_NT_local.op.reduce_axis)
    T_batch_matmul_NT_local_b_c_o_i, T_batch_matmul_NT_local_b_c_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_b_c, factor=1)
    T_batch_matmul_NT_local_b_c_o_o_i, T_batch_matmul_NT_local_b_c_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_b_c_o_i, factor=1)
    T_batch_matmul_NT_local_b_c_o_o_o_i, T_batch_matmul_NT_local_b_c_o_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_b_c_o_o_i, factor=1)
    T_batch_matmul_NT_local_b_c_o_o_o_o, T_batch_matmul_NT_local_b_c_o_o_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_b_c_o_o_o_i, factor=1)
    T_batch_matmul_NT_local_i_c_o_i, T_batch_matmul_NT_local_i_c_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_i_c, factor=4)
    T_batch_matmul_NT_local_i_c_o_o_i, T_batch_matmul_NT_local_i_c_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_i_c_o_i, factor=2)
    T_batch_matmul_NT_local_i_c_o_o_o_i, T_batch_matmul_NT_local_i_c_o_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_i_c_o_o_i, factor=8)
    T_batch_matmul_NT_local_i_c_o_o_o_o, T_batch_matmul_NT_local_i_c_o_o_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_i_c_o_o_o_i, factor=2)
    T_batch_matmul_NT_local_j_c_o_i, T_batch_matmul_NT_local_j_c_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_j_c, factor=2)
    T_batch_matmul_NT_local_j_c_o_o_i, T_batch_matmul_NT_local_j_c_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_j_c_o_i, factor=1)
    T_batch_matmul_NT_local_j_c_o_o_o_i, T_batch_matmul_NT_local_j_c_o_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_j_c_o_o_i, factor=16)
    T_batch_matmul_NT_local_j_c_o_o_o_o, T_batch_matmul_NT_local_j_c_o_o_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_j_c_o_o_o_i, factor=1)
    T_batch_matmul_NT_local_k_o_i, T_batch_matmul_NT_local_k_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_k, factor=1)
    T_batch_matmul_NT_local_k_o_o, T_batch_matmul_NT_local_k_o_i = s[T_batch_matmul_NT_local].split(T_batch_matmul_NT_local_k_o_i, factor=32)
    s[T_batch_matmul_NT_local].reorder(T_batch_matmul_NT_local_b_c_o_o_o_o, T_batch_matmul_NT_local_i_c_o_o_o_o, T_batch_matmul_NT_local_j_c_o_o_o_o, T_batch_matmul_NT_local_b_c_o_o_o_i, T_batch_matmul_NT_local_i_c_o_o_o_i, T_batch_matmul_NT_local_j_c_o_o_o_i, T_batch_matmul_NT_local_b_c_o_o_i, T_batch_matmul_NT_local_i_c_o_o_i, T_batch_matmul_NT_local_j_c_o_o_i, T_batch_matmul_NT_local_k_o_o, T_batch_matmul_NT_local_k_o_i, T_batch_matmul_NT_local_b_c_o_i, T_batch_matmul_NT_local_i_c_o_i, T_batch_matmul_NT_local_j_c_o_i, T_batch_matmul_NT_local_k_i, T_batch_matmul_NT_local_b_c_i, T_batch_matmul_NT_local_i_c_i, T_batch_matmul_NT_local_j_c_i)
    T_batch_matmul_NT_b_o_i, T_batch_matmul_NT_b_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_b, factor=1)
    T_batch_matmul_NT_b_o_o_i, T_batch_matmul_NT_b_o_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_b_o_i, factor=1)
    T_batch_matmul_NT_b_o_o_o, T_batch_matmul_NT_b_o_o_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_b_o_o_i, factor=1)
    T_batch_matmul_NT_i_o_i, T_batch_matmul_NT_i_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_i, factor=8)
    T_batch_matmul_NT_i_o_o_i, T_batch_matmul_NT_i_o_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_i_o_i, factor=8)
    T_batch_matmul_NT_i_o_o_o, T_batch_matmul_NT_i_o_o_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_i_o_o_i, factor=2)
    T_batch_matmul_NT_j_o_i, T_batch_matmul_NT_j_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_j, factor=2)
    T_batch_matmul_NT_j_o_o_i, T_batch_matmul_NT_j_o_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_j_o_i, factor=16)
    T_batch_matmul_NT_j_o_o_o, T_batch_matmul_NT_j_o_o_i = s[T_batch_matmul_NT].split(T_batch_matmul_NT_j_o_o_i, factor=1)
    s[T_batch_matmul_NT].reorder(T_batch_matmul_NT_b_o_o_o, T_batch_matmul_NT_i_o_o_o, T_batch_matmul_NT_j_o_o_o, T_batch_matmul_NT_b_o_o_i, T_batch_matmul_NT_i_o_o_i, T_batch_matmul_NT_j_o_o_i, T_batch_matmul_NT_b_o_i, T_batch_matmul_NT_i_o_i, T_batch_matmul_NT_j_o_i, T_batch_matmul_NT_b_i, T_batch_matmul_NT_i_i, T_batch_matmul_NT_j_i)
    s[T_batch_matmul_NT_local].compute_at(s[T_batch_matmul_NT], T_batch_matmul_NT_j_o_i)
    B_shared = s.cache_read(B, "shared", [T_batch_matmul_NT_local])
    B_shared_ax0, B_shared_ax1, B_shared_ax2 = tuple(B_shared.op.axis)
    s[B_shared].compute_at(s[T_batch_matmul_NT_local], T_batch_matmul_NT_local_k_o_o)
    A_shared = s.cache_read(A, "shared", [T_batch_matmul_NT_local])
    A_shared_ax0, A_shared_ax1, A_shared_ax2 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[T_batch_matmul_NT_local], T_batch_matmul_NT_local_k_o_o)
    T_batch_matmul_NT_b_o_o_o_i_o_o_o_fused_j_o_o_o_fused = s[T_batch_matmul_NT].fuse(T_batch_matmul_NT_b_o_o_o, T_batch_matmul_NT_i_o_o_o, T_batch_matmul_NT_j_o_o_o)
    s[T_batch_matmul_NT].bind(T_batch_matmul_NT_b_o_o_o_i_o_o_o_fused_j_o_o_o_fused, te.thread_axis("blockIdx.x"))
    T_batch_matmul_NT_b_o_o_i_i_o_o_i_fused_j_o_o_i_fused = s[T_batch_matmul_NT].fuse(T_batch_matmul_NT_b_o_o_i, T_batch_matmul_NT_i_o_o_i, T_batch_matmul_NT_j_o_o_i)
    s[T_batch_matmul_NT].bind(T_batch_matmul_NT_b_o_o_i_i_o_o_i_fused_j_o_o_i_fused, te.thread_axis("vthread"))
    T_batch_matmul_NT_b_o_i_i_o_i_fused_j_o_i_fused = s[T_batch_matmul_NT].fuse(T_batch_matmul_NT_b_o_i, T_batch_matmul_NT_i_o_i, T_batch_matmul_NT_j_o_i)
    s[T_batch_matmul_NT].bind(T_batch_matmul_NT_b_o_i_i_o_i_fused_j_o_i_fused, te.thread_axis("threadIdx.x"))
    B_shared_ax0_ax1_fused_ax2_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1, B_shared_ax2)
    B_shared_ax0_ax1_fused_ax2_fused_o, B_shared_ax0_ax1_fused_ax2_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused_ax2_fused, factor=1)
    s[B_shared].vectorize(B_shared_ax0_ax1_fused_ax2_fused_i)
    B_shared_ax0_ax1_fused_ax2_fused_o_o, B_shared_ax0_ax1_fused_ax2_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_ax2_fused_o, factor=128)
    s[B_shared].bind(B_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))
    A_shared_ax0_ax1_fused_ax2_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1, A_shared_ax2)
    A_shared_ax0_ax1_fused_ax2_fused_o, A_shared_ax0_ax1_fused_ax2_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused_ax2_fused, factor=2)
    s[A_shared].vectorize(A_shared_ax0_ax1_fused_ax2_fused_i)
    A_shared_ax0_ax1_fused_ax2_fused_o_o, A_shared_ax0_ax1_fused_ax2_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_ax2_fused_o, factor=128)
    s[A_shared].bind(A_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))
    s[T_batch_matmul_NT_local].pragma(T_batch_matmul_NT_local_b_c_o_o_o_o, "auto_unroll_max_step", 16)
    s[T_batch_matmul_NT_local].pragma(T_batch_matmul_NT_local_b_c_o_o_o_o, "unroll_explicit", True)

    return s, args
#--------------------------------------------------------------------------
def softmax_layer_schedule(batch_size, target, cu_save=False):
    args = softmax_layer(batch_size, 1, 1000)
    s    = te.create_schedule(args[1].op)

    # ========== Task (workload key: ["softmax_layer", 128, 1, 1000]) ==========
    # data = PLACEHOLDER [128, 1, 1000]
    # T_softmax_maxelem(i0, i1) max= data[i0, i1, k]
    # T_softmax_exp(i0, i1, i2) = tir.exp((data[i0, i1, i2] - T_softmax_maxelem[i0, i1]))
    # T_softmax_expsum(i0, i1) += T_softmax_exp[i0, i1, k]
    # T_softmax_norm(i0, i1, i2) = (T_softmax_exp[i0, i1, i2]/T_softmax_expsum[i0, i1])

    # Parameters
    T_softmax_norm    = args[1]                                     # args[1]
    T_softmax_exp     = T_softmax_norm.op.body[0].a.producer
    T_softmax_expsum  = T_softmax_norm.op.body[0].b.producer
    T_softmax_maxelem = T_softmax_exp.op.body[0].args[0].b.producer
    data              = T_softmax_exp.op.body[0].args[0].a.producer # args[0]

    assert(args[1] == T_softmax_norm and
           args[0] == data)

    T_softmax_maxelem_i0, T_softmax_maxelem_i1, T_softmax_maxelem_k = tuple(T_softmax_maxelem.op.axis) + tuple(T_softmax_maxelem.op.reduce_axis)
    T_softmax_exp_i0, T_softmax_exp_i1, T_softmax_exp_i2 = tuple(T_softmax_exp.op.axis) + tuple(T_softmax_exp.op.reduce_axis)
    T_softmax_expsum_i0, T_softmax_expsum_i1, T_softmax_expsum_k = tuple(T_softmax_expsum.op.axis) + tuple(T_softmax_expsum.op.reduce_axis)
    T_softmax_norm_i0, T_softmax_norm_i1, T_softmax_norm_i2 = tuple(T_softmax_norm.op.axis) + tuple(T_softmax_norm.op.reduce_axis)
    T_softmax_norm_i2_o, T_softmax_norm_i2_i = s[T_softmax_norm].split(T_softmax_norm_i2, factor=64)
    s[T_softmax_norm].bind(T_softmax_norm_i2_i, te.thread_axis("threadIdx.x"))
    T_softmax_expsum_k_o, T_softmax_expsum_k_i = s[T_softmax_expsum].split(T_softmax_expsum_k, factor=64)
    s[T_softmax_expsum].bind(T_softmax_expsum_k_i, te.thread_axis("threadIdx.x"))
    s[T_softmax_expsum].compute_at(s[T_softmax_norm], T_softmax_norm_i1)
    s[T_softmax_exp].compute_inline()
    T_softmax_maxelem_k_o, T_softmax_maxelem_k_i = s[T_softmax_maxelem].split(T_softmax_maxelem_k, factor=64)
    s[T_softmax_maxelem].bind(T_softmax_maxelem_k_i, te.thread_axis("threadIdx.x"))
    s[T_softmax_maxelem].compute_at(s[T_softmax_norm], T_softmax_norm_i1)
    T_softmax_norm_i0_i1_fused = s[T_softmax_norm].fuse(T_softmax_norm_i0, T_softmax_norm_i1)
    s[T_softmax_norm].bind(T_softmax_norm_i0_i1_fused, te.thread_axis("blockIdx.x"))
    s[T_softmax_maxelem].pragma(T_softmax_maxelem_i0, "auto_unroll_max_step", 16)
    s[T_softmax_maxelem].pragma(T_softmax_maxelem_i0, "unroll_explicit", True)
    s[T_softmax_expsum].pragma(T_softmax_expsum_i0, "auto_unroll_max_step", 0)
    s[T_softmax_expsum].pragma(T_softmax_expsum_i0, "unroll_explicit", True)

    return s, args
#--------------------------------------------------------------------------
def pool2d_layer_schedule(batch_size, target, cu_save=False):
    args = pool2d_layer(batch_size, 56, 56, 64, 3, 3, 2, 0)
    s    = te.create_schedule(args[1].op)

    # ========== Task (workload key: ["pool2d_layer", 128, 56, 56, 64, 3, 3, 2, 0]) ==========
    # data = PLACEHOLDER [128, 56, 56, 64]
    # pool_max(ax0, ax1, ax2, ax3) max= data[ax0, ((ax1*2) + rv0), ((ax2*2) + rv1), ax3]

    # Parameters
    pool_max = args[1]                                # args[1]
    data     = pool_max.op.body[0].source[0].producer # args[0]

    assert(args[1] == pool_max and
           args[0] == data)

    pool_max_ax0, pool_max_ax1, pool_max_ax2, pool_max_ax3, pool_max_rv0, pool_max_rv1 = tuple(pool_max.op.axis) + tuple(pool_max.op.reduce_axis)
    pool_max_ax0_ax1_fused_ax2_fused_ax3_fused = s[pool_max].fuse(pool_max_ax0, pool_max_ax1, pool_max_ax2, pool_max_ax3)
    pool_max_ax0_ax1_fused_ax2_fused_ax3_fused_o, pool_max_ax0_ax1_fused_ax2_fused_ax3_fused_i = s[pool_max].split(pool_max_ax0_ax1_fused_ax2_fused_ax3_fused, factor=64)
    s[pool_max].bind(pool_max_ax0_ax1_fused_ax2_fused_ax3_fused_o, te.thread_axis("blockIdx.x"))
    s[pool_max].bind(pool_max_ax0_ax1_fused_ax2_fused_ax3_fused_i, te.thread_axis("threadIdx.x"))
    s[pool_max].pragma(pool_max_ax0_ax1_fused_ax2_fused_ax3_fused_o, "auto_unroll_max_step", 1024)
    s[pool_max].pragma(pool_max_ax0_ax1_fused_ax2_fused_ax3_fused_o, "unroll_explicit", True)

    return s, args
#--------------------------------------------------------------------------

################################################################################
# Auto-tuning and Evaluate layers (conv2d, bgemm, batch_norm, softmax)
# ----------------------------------

for batch_size in [128]:
    target           = tvm.target.Target("cuda")
    dtype            = "float32"

    # Auto-tuning log file paths
    conv2d_log_file  = "json/conv2d_B%d-%s.json" % (batch_size, target.kind.name)
    bgemm_log_file   = "json/bgemm_B%d-%s.json" % (batch_size, target.kind.name)
    softmax_log_file = "json/softmax_B%d-%s.json" % (batch_size, target.kind.name)
    pool2d_log_file  = "json/pool2d_B%d-%s.json" % (batch_size, target.kind.name)

    # Kernel code paths
    conv2d_cu_file   = "cuda/conv2d_B%d.cu" % (batch_size)
    bgemm_cu_file    = "cuda/bgemm_B%d.cu" % (batch_size)
    softmax_cu_file  = "cuda/softmax_B%d.cu" % (batch_size)
    pool2d_cu_file   = "cuda/pool2d_B%d.cu" % (batch_size)

    # Modified kernel code paths
    conv2d_modified_cu_file  = "cuda_modified/conv2d_B%s.cu" % (batch_size)
    bgemm_modified_cu_file   = "cuda_modified/bgemm_B%s.cu" % (batch_size)
    softmax_modified_cu_file = "cuda_modified/softmax_B%s.cu" % (batch_size)
    pool2d_modified_cu_file  = "cuda_modified/pool2d_B%s.cu" % (batch_size)

    ###############################################
    # 1. Auto-tune & Analyze the layers
    ###############################################

    # conv2d layer
    conv2d_task = conv2d_layer_tuning(batch_size, target=target, log_file=conv2d_log_file)
    analyze_task(conv2d_task, conv2d_log_file)

    # bgemm layer
    bgemm_task = bgemm_layer_tuning(batch_size, target=target, log_file=bgemm_log_file)
    analyze_task(bgemm_task, bgemm_log_file)

    # softmax layer
    softmax_task = softmax_layer_tuning(batch_size, target=target, log_file=softmax_log_file)
    analyze_task(softmax_task, softmax_log_file)

    # pool2d layer
    pool2d_task = pool2d_layer_tuning(batch_size, target=target, log_file=pool2d_log_file)
    analyze_task(pool2d_task, pool2d_log_file)

    ###############################################
    # 2. Adjust the TIR scheduling
    ###############################################

    conv2d_sch, conv2d_args   = conv2d_layer_schedule(batch_size, target)
    bgemm_sch, bgemm_args     = bgemm_layer_schedule(batch_size, target)
    softmax_sch, softmax_args = softmax_layer_schedule(batch_size, target, cu_save=True)
    pool2d_sch, pool2d_args   = pool2d_layer_schedule(batch_size, target)

    ###############################################
    # 3. Analyze the schedule
    ###############################################

    analyze_sch(conv2d_sch, conv2d_args, target, cu_file=conv2d_modified_cu_file)
    analyze_sch(bgemm_sch, bgemm_args, target, cu_file=bgemm_modified_cu_file)
    analyze_sch(softmax_sch, softmax_args, target, cu_file=softmax_modified_cu_file)
    analyze_sch(pool2d_sch, pool2d_args, target, cu_file=pool2d_modified_cu_file)

    ###############################################
    # 4. Evaluate the modified layers
    ###############################################

    conv2d_layer_eval(batch_size, conv2d_sch, conv2d_args, target=target)
    bgemm_layer_eval(batch_size, bgemm_sch, bgemm_args, target=target)
    softmax_layer_eval(batch_size, softmax_sch, softmax_args, target=target)
    pool2d_layer_eval(batch_size, pool2d_sch, pool2d_args, target=target)