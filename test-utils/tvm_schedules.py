import tvm
from tvm import relay, te, topi
from tvm import auto_scheduler
#-----------------------------------------------------------------------------------------------
def cuda_schedule_bgemm(batch_size, M, K, N):
    A   = te.placeholder((batch_size, M, K), name="A")
    B   = te.placeholder((batch_size, N, K), name="B")

    with tvm.target.Target("cuda"):
        out = topi.cuda.batch_matmul(A, B, (batch_size, M, N), "float32", False, True)
        sst = topi.cuda.schedule_batch_matmul([out])

    return sst, [A, B, out]
#-----------------------------------------------------------------------------------------------
def cuda_schedule_conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data   = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    with tvm.target.Target("cuda"):
        out = topi.cuda.conv2d_nchw(data, kernel, stride, padding, 1 ,"float32")
        sst = topi.cuda.schedule_conv2d_nchw([out])

    return sst, [data, kernel, out]
#-----------------------------------------------------------------------------------------------
@auto_scheduler.register_workload
def bgemm_workload(batch_size, M, K, N):
    A   = te.placeholder((batch_size, M, K), name="A")
    B   = te.placeholder((batch_size, N, K), name="B")

    with tvm.target.Target("cuda"):
        out = topi.cuda.batch_matmul(A, B, (batch_size, M, N), "float32", False, True)

    return A, B, out
#-----------------------------------------------------------------------------------------------
@auto_scheduler.register_workload
def conv2d_workload(N, H, W, CO, CI, KH, KW, stride, padding):
    data   = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    with tvm.target.Target("cuda"):
        out = topi.cuda.conv2d_nchw(data, kernel, stride, padding, 1 ,"float32")

    return data, kernel, out
#-----------------------------------------------------------------------------------------------
def get_bgemm_shape(batch_size, M, K, N):
    bgemm_A_shape      = (batch_size, M, K)
    bgemm_B_shape      = (batch_size, N, K)
    bgemm_output_shape = (batch_size, M, N)
    
    return (bgemm_A_shape, bgemm_B_shape, bgemm_output_shape)
#-----------------------------------------------------------------------------------------------
def get_conv2d_shape(N, H, W, CO, CI, KH, KW, stride, padding):
    conv2d_data_shape   = (N, CI, H, W)
    conv2d_kernel_shape = (CO, CI, KH, KW)
    conv2d_output_shape = (N, CO,
                           int((H - KH + 2 * padding) / stride) + 1,
                           int((W - KW + 2 * padding) / stride) + 1)
    
    return (conv2d_data_shape, conv2d_kernel_shape, conv2d_output_shape)
#-----------------------------------------------------------------------------------------------