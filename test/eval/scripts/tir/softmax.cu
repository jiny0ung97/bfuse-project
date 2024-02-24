# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(data: T.Buffer((128, 1, 1000), "float32"), T_softmax_norm: T.Buffer((128, 1, 1000), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        T_softmax_maxelem = T.allocate([128], "float32", "global")
        T_softmax_exp = T.allocate([128000], "float32", "global")
        T_softmax_maxelem_1 = T.Buffer((128,), data=T_softmax_maxelem)
        data_1 = T.Buffer((128000,), data=data.data)
        with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            T_softmax_maxelem_1[threadIdx_x] = T.float32(-3.4028234663852886e+38)
            for k in range(1000):
                T_softmax_maxelem_1[threadIdx_x] = T.max(T_softmax_maxelem_1[threadIdx_x], data_1[threadIdx_x * 1000 + k])
        T_softmax_exp_1 = T.Buffer((128000,), data=T_softmax_exp)
        with T.launch_thread("blockIdx.x", 125) as blockIdx_x:
            threadIdx_x = T.launch_thread("threadIdx.x", 1024)
            T_softmax_exp_1[blockIdx_x * 1024 + threadIdx_x] = T.exp(data_1[blockIdx_x * 1024 + threadIdx_x] - T_softmax_maxelem_1[(blockIdx_x * 128 + threadIdx_x // 8) // 125])
        T_softmax_maxelem_2 = T.Buffer((128,), data=T_softmax_maxelem)
        with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            T_softmax_maxelem_2[threadIdx_x] = T.float32(0)
            for k in range(1000):
                T_softmax_maxelem_2[threadIdx_x] = T_softmax_maxelem_2[threadIdx_x] + T_softmax_exp_1[threadIdx_x * 1000 + k]
        blockIdx_x = T.launch_thread("blockIdx.x", 125)
        threadIdx_x = T.launch_thread("threadIdx.x", 1024)
        T_softmax_norm_1 = T.Buffer((128000,), data=T_softmax_norm.data)
        T_softmax_norm_1[blockIdx_x * 1024 + threadIdx_x] = T_softmax_exp_1[blockIdx_x * 1024 + threadIdx_x] / T_softmax_maxelem_2[(blockIdx_x * 128 + threadIdx_x // 8) // 125]