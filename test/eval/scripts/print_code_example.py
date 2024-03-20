import sys
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

#################################################################
# Define a Network
# ----------------

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
    return mod, params, input_shape, output_shape


# Define the neural network and compilation target
network = "resnet-18"
batch_size = 1
layout = "NHWC"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

#################################################################
# Print Relay
# --------------------

# Print Relay from the network
print("Get model...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

#################################################################
# Print TIR
# --------------------

# Print TIR Pass
@tvm.tir.transform.prim_func_pass(opt_level=0)
def dump_tir(f, mod, ctx):
    print(f)
    return f

# adjust the desired level of optimization
opt_level = 0
target = tvm.target.cuda()

# Compile the model
print("Compile the model...")
with tvm.transform.PassContext(opt_level=opt_level, config={"tir.add_lower_pass": [(0, dump_tir)]}):
    lib = relay.build(mod, target, params=params)

#################################################################
# Print CUDA Kernel
# --------------------

# Print CUDA Kernel
print("Print CUDA Kernel code...")
print(lib.lib.imported_modules[0].get_source())