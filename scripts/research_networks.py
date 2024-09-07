#!/usr/bin/python3

import tvm
from tvm import relay
import tvm.relay.testing

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch, onnx

#-------------------------------------------------------------------------------#
def download_network(name, batch_size, seq_len=512, dtype="float32"):
    # Settings
    access_token = "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"

    # login huggingface
    login(access_token)

    # Input data
    inputs = torch.ones(batch_size, seq_len, dtype=torch.int64)

    if name.startswith("Meta-Llama-"):
        # meta-llama/Meta-Llama-3-8B
        tokenizer  = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        model      = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float32)
        model_name = "Meta-Llama-3-8B-%d-%d.onnx" % (batch_size, seq_len)
    elif name.startswith("gpt"):
        # openai-community/gpt2
        tokenizer  = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model      = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.float32)
        model_name = "gpt2-%d-%d.onnx" % (batch_size, seq_len)
    elif name.startswith("bert-"):
        # google-bert/bert-base-uncased
        tokenizer  = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model      = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float32)
        model_name = "bert-base-uncased-%d-%d.onnx" % (batch_size, seq_len)

    torch.onnx.export(model, inputs, model_name, False, input_names=["input_ids"])
#-------------------------------------------------------------------------------#
def get_network(name, batch_size, seq_len=512, layout="NHWC", dtype="float32"):
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
    elif name.startswith("Meta-Llama-") or name.startswith("gpt") or name.startswith("bert-"):
        onnx_path  = "%s-%d-%d.onnx" % (name, batch_size, seq_len)
        onnx_model = onnx.load(onnx_path)

        input_name = "input_ids"
        shape_dict = {input_name: (batch_size, seq_len)}
        mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape_dict)
    # elif name.startswith("unet-"):
    #     n_layer = int(name.split("-")[1])
    #     shape   = (1,1,256,256)
    #     input0  = torch.randn(shape).half().cuda()
    #     path  = 'unet2d_fp16.trace'
    #     trace = torch.jit.load(path).half().cuda()
    #     mod, params = relay.frontend.from_pytorch(trace, [('input0',input0.shape)], default_dtype='float16')

    return mod, params, input_shape, output_shape
#-------------------------------------------------------------------------------#
def main():

    # Define the neural network and compilation target
    network    = "resnet-18"
    # network    = "bert-base-uncased"
    # network    = "gpt2"
    # network    = "Meta-Llama-3-8B"
    # batch_size = 1024
    batch_size = 1
    layout     = "NCHW"
    target     = tvm.target.Target("cuda")
    dtype      = "float32"
    file       = "%s.txt" % (network)
    seq_len    = 512

    #################################################################
    # Download LLM Models
    # --------------------    
    # download_network(network, batch_size)

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
    mod, params, input_shape, output_shape = get_network(network, batch_size, seq_len, layout, dtype=dtype)

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