# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(data: T.Buffer((128, 64, 56, 56), "float32"), pool_max: T.Buffer((128, 64, 27, 27), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        blockIdx_x = T.launch_thread("blockIdx.x", 5832)
        pool_max_local = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 1024)
        pool_max_local_1 = T.Buffer((1,), data=pool_max_local, scope="local", align=4)
        pool_max_local_1[0] = T.float32(-3.4028234663852886e+38)
        for rv0, rv1 in T.grid(3, 3):
            data_1 = T.Buffer((25690112,), data=data.data)
            pool_max_local_1[0] = T.max(pool_max_local_1[0], data_1[(blockIdx_x * 1024 + threadIdx_x) // 729 * 3136 + (blockIdx_x * 295 + threadIdx_x) % 729 // 27 * 112 + rv0 * 56 + (blockIdx_x * 25 + threadIdx_x) % 27 * 2 + rv1])
        pool_max_1 = T.Buffer((5971968,), data=pool_max.data)
        pool_max_1[blockIdx_x * 1024 + threadIdx_x] = pool_max_local_1[0]