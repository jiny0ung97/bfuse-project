
import numpy as np
import mxnet as mx

import tvm
from tvm import relay
from tvm import te, topi
from tvm.relay import testing
#--------------------------------------------------------------------------
@tvm.tir.transform.prim_func_pass(opt_level=0)
def dump_tir(f, mod, ctx):
    # dump tir expressions
    print(f)

    return f
#--------------------------------------------------------------------------
def conv2d_from_relay(target, dtype, cu_file):
    data_shape   = (128, 56, 56, 64)
    weight_shape = (3, 3, 64, 128)

    data   = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)

    out = relay.nn.conv2d(
        data=data, weight=weight, padding=(1, 1), data_layout="NHWC", kernel_layout="HWIO"
    )

    net = relay.Function(relay.analysis.free_vars(out), out)
    # mod = tvm.IRModule.from_expr(net)
    # print(mod)

    # mod = relay.transform.InferType()(mod)
    # mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)

    net, params = testing.create_workload(net)
    with tvm.transform.PassContext(opt_level=3, config={"tir.add_lower_pass": [(3, dump_tir)]}):
        lib = relay.build(net, target=target, params=params)

    # Print CUDA code
    print("Get CUDA kernel code...")
    with open(cu_file, "w+") as f:
        f.write(lib.lib.imported_modules[0].get_source())
#--------------------------------------------------------------------------
def conv2d_from_topi(target, dtype, cu_file):
    data_shape   = (128, 64, 56, 56)
    weight_shape = (64, 128, 3, 3)

    data   = te.placeholder(data_shape, name="data")
    kernel = te.placeholder(weight_shape, name="kernel")
    # out    = topi.cuda.conv2d_nchw(data, kernel, stride=1, padding=1, dilation=1, out_dtype="float32")
    out    = topi.cuda.conv2d_nchw(data, kernel, 1, 1, 1, "float32")

    sch  = te.create_schedule(out.op)
    args = [data, kernel, out]

    # Print lowered tir code
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    print("Get CUDA kernel code...")
    with open(cu_file, "w+") as f:
        mod = tvm.build(sch, args, target=target)
        f.write(mod.imported_modules[0].get_source())
#--------------------------------------------------------------------------
def conv2d_from_document(target, dtype, cu_file):
    # The sizes of inputs and filters
    batch = 256
    in_channel = 256
    out_channel = 512
    in_size = 14
    kernel = 3
    pad = 1
    stride = 1

    # Algorithm
    A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
    W = te.placeholder((kernel, kernel, in_channel, out_channel), name="W")
    out_size = (in_size - kernel + 2 * pad) // stride + 1 # 14
    # Pad input
    Apad = te.compute(
        (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.tir.if_then_else(
            tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )
    # Create reduction variables
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    # Compute the convolution
    B = te.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: te.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
        ),
        name="B",
    )

    # Designate the memory hierarchy
    s = te.create_schedule(B.op)
    s[Apad].compute_inline()  # compute Apad inline
    AA = s.cache_read(Apad, "shared", [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    # tile consts
    tile = 8
    num_thread = 16
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    by, fi = s[B].split(fi, factor=block_factor)
    bx, ni = s[B].split(ni, factor=block_factor)

    # Bind the iteration variables to GPU thread indices
    # s[B].bind(bz, block_z)
    # s[B].bind(by, block_y)
    # s[B].bind(bx, block_x)

    tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, ni = s[B].split(ni, nparts=num_thread)
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    # s[B].bind(ty, thread_y)
    # s[B].bind(tx, thread_x)

    # Hoist this schedule
    s[BL].compute_at(s[B], tx)

    # Fuse the iterations
    b = s[B].fuse(bz, by, bx)
    t = s[B].fuse(ty, tx)
    s[B].bind(b, block_x)
    s[B].bind(t, thread_x)

    # Schedule BL local write
    # s[BL].compute_at(s[B], tx)
    yi, xi, fi, ni = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    s[BL].reorder(rco, ry, rx, rci, fi, ni)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    # s[AA].bind(ty, thread_y)
    # s[AA].bind(tx, thread_x)
    t = s[AA].fuse(ty, tx)
    s[AA].bind(t, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    yi, xi, ci, fi = s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    _, fi = s[WW].split(fi, factor=4)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    # s[WW].bind(ty, thread_y)
    # s[WW].bind(tx, thread_x)
    t = s[WW].fuse(ty, tx)
    s[WW].bind(t, thread_x)
    s[WW].vectorize(fi)  # vectorize memory load

    # Print lowered tir code
    print("Lowered TIR:")
    print(tvm.lower(s, [A, W, B], simple_mode=True))

    # Print CUDA code
    func = tvm.build(s, [A, W, B], target=target)
    print("Get CUDA kernel code...")
    with open(cu_file, "w+") as f:
        f.write(func.imported_modules[0].get_source())
#--------------------------------------------------------------------------

################################################################################
# Get default CUDA kernel code (conv2d, bgemm)
# ----------------------------------

target  = tvm.target.Target("cuda")
dtype   = "float32"

relay_cu_file    = "default/conv2d_relay.cu"
topi_cu_file     = "default/conv2d_topi.cu"
document_cu_file = "default/conv2d_document.cu"

# conv2d_from_relay(target, dtype, relay_cu_file)
# conv2d_from_topi(target, dtype, topi_cu_file)
conv2d_from_document(target, dtype, document_cu_file)