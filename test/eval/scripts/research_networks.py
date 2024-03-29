
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

#-------------------------------------------------------------------------------#
def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name.startswith("densenet-"):
        assert layout == "NCHW", "densenet only supports NCHW layout"
        densnet_size = int(name.split("-")[1])
        mod, params = relay.testing.densenet.get_workload(
            densenet_size=densnet_size,
            classes=1000,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("unet-"):
        n_layer = int(name.split("-")[1])
        shape   = (1,1,256,256)
        input0  = torch.randn(shape).half().cuda()
        path  = 'unet2d_fp16.trace'
        trace = torch.jit.load(path).half().cuda()
        mod, params = relay.frontend.from_pytorch(trace, [('input0',input0.shape)], default_dtype='float16')
    return mod, params, input_shape, output_shape
#-------------------------------------------------------------------------------#
# Main Function
def main():
    # Define the neural network and compilation target
    network    = "mobilenet"
    batch_size = 1
    layout     = "NCHW"
    target     = tvm.target.Target("cuda")
    dtype      = "float32"
    file       = "network/%s.txt" % (network)

    #################################################################
    # Extract Search Tasks
    # --------------------
    # Next, we extract the search tasks and their weights from a network.
    # The weight of a task is the number of appearances of the task's subgraph
    # in the whole network.
    # By using the weight, we can approximate the end-to-end latency of the network
    # as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
    # latency of a task and :code:`weight[t]` is the weight of the task.
    # The task scheduler will just optimize this objective.

    # Extract tasks from the network
    print("Extract tasks...")
    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

    with open(file, "w+") as f:
        f.write(str(mod))

    # tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    # with open(file, "w+") as f:
    #     for idx, task in enumerate(tasks):
    #         f.write("\n========== Task %d  (workload key: %s) ==========\n" % (idx, task.workload_key))
    #         f.write(str(task.compute_dag))
#-------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
#-------------------------------------------------------------------------------#    