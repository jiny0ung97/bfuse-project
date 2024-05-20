import os, sys
import logging

tvm_kernels_module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../tvm-kernels")
sys.path.append(tvm_kernels_module_path)

import tvm_schedules
#-----------------------------------------------------------------------------------------------
def get_shape_decl_str(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]
    kernel_info = infoYAML["KernelInfo"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    shape_decl_str = ""

    # Get kernel1's shape declarations
    for idx, kname in enumerate(fusion_sets[0]["Set"]):
        kargs = kernel_info[kname]

        if kname.startswith("bgemm"):
            shape = tvm_schedules.get_bgemm_shape(*kargs)

            shape_decl_str += f"static int kernel1_I_shape_{idx}[] = {{{shape[0][0]}, {shape[0][1]}, {shape[0][2]}, 1}};\n"
            shape_decl_str += f"static int kernel1_F_shape_{idx}[] = {{{shape[1][0]}, {shape[1][1]}, {shape[1][2]}, 1}};\n"
            shape_decl_str += f"static int kernel1_O_shape_{idx}[] = {{{shape[2][0]}, {shape[2][1]}, {shape[2][2]}, 1}};\n"
        elif kname.startswith("conv2d"):
            shape = tvm_schedules.get_conv2d_shape(*kargs)

            shape_decl_str += f"static int kernel1_I_shape_{idx}[] = {{{shape[0][0]}, {shape[0][1]}, {shape[0][2]}, {shape[0][3]}}};\n"
            shape_decl_str += f"static int kernel1_F_shape_{idx}[] = {{{shape[1][0]}, {shape[1][1]}, {shape[1][2]}, {shape[1][3]}}};\n"
            shape_decl_str += f"static int kernel1_O_shape_{idx}[] = {{{shape[2][0]}, {shape[2][1]}, {shape[2][2]}, {shape[2][3]}}};\n"
        else:
            logging.ERROR("Unknown kernel's type.")
            exit(0)

    shape_decl_str += "//------------------------------------------------------------------------------------------\n"

    # Get kernel2's shape declarations
    for idx, kname in enumerate(fusion_sets[1]["Set"]):
        kargs = kernel_info[kname]

        if kname.startswith("bgemm"):
            shape = tvm_schedules.get_bgemm_shape(*kargs)

            shape_decl_str += f"static int kernel2_I_shape_{idx}[] = {{{shape[0][0]}, {shape[0][1]}, {shape[0][2]}, 1}};\n"
            shape_decl_str += f"static int kernel2_F_shape_{idx}[] = {{{shape[1][0]}, {shape[1][1]}, {shape[1][2]}, 1}};\n"
            shape_decl_str += f"static int kernel2_O_shape_{idx}[] = {{{shape[2][0]}, {shape[2][1]}, {shape[2][2]}, 1}};\n"
        elif kname.startswith("conv2d"):
            shape = tvm_schedules.get_conv2d_shape(*kargs)

            shape_decl_str += f"static int kernel2_I_shape_{idx}[] = {{{shape[0][0]}, {shape[0][1]}, {shape[0][2]}, {shape[0][3]}}};\n"
            shape_decl_str += f"static int kernel2_F_shape_{idx}[] = {{{shape[1][0]}, {shape[1][1]}, {shape[1][2]}, {shape[1][3]}}};\n"
            shape_decl_str += f"static int kernel2_O_shape_{idx}[] = {{{shape[2][0]}, {shape[2][1]}, {shape[2][2]}, {shape[2][3]}}};\n"
        else:
            logging.ERROR("Unknown kernel's type.")
            exit(0)

    shape_decl_str += "//------------------------------------------------------------------------------------------\n"

    return shape_decl_str
#-----------------------------------------------------------------------------------------------
def get_dim_decl_str(infoYAML, kernelsYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    dim_decl_str = ""

    # Get kernel1's dimension declarations
    for idx, kname in enumerate(fusion_sets[0]["Set"]):
        gridDim  = kernelsYAML[kname]["GridDim"]
        blockDim = kernelsYAML[kname]["BlockDim"]

        dim_decl_str += f"static unsigned int kernel1_gridDim_{idx}[] = {{{gridDim['X']}, {gridDim['Y']}, {gridDim['Z']}}};\n"
        dim_decl_str += f"static unsigned int kernel1_blockDim_{idx}[] = {{{blockDim['X']}, {blockDim['Y']}, {blockDim['Z']}}};\n"

    dim_decl_str += "//------------------------------------------------------------------------------------------\n"

    # Get kernel2's dimension declarations
    for idx, kname in enumerate(fusion_sets[1]["Set"]):
        gridDim  = kernelsYAML[kname]["GridDim"]
        blockDim = kernelsYAML[kname]["BlockDim"]

        dim_decl_str += f"static unsigned int kernel2_gridDim_{idx}[] = {{{gridDim['X']}, {gridDim['Y']}, {gridDim['Z']}}};\n"
        dim_decl_str += f"static unsigned int kernel2_blockDim_{idx}[] = {{{blockDim['X']}, {blockDim['Y']}, {blockDim['Z']}}};\n"

    dim_decl_str += "//------------------------------------------------------------------------------------------\n"

    return dim_decl_str
#-----------------------------------------------------------------------------------------------
def get_fuse_dim_decl_str(infoYAML, hfuseYAML, bfuseYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    hfuse_kernels = []
    bfuse_kernels = []

    # Get fused kernel names
    for kname1 in fusion_sets[0]["Set"]:
        for kname2 in fusion_sets[1]["Set"]:
            hfuse_kernels.append("%s_%s_fused_hfuse" % (kname1, kname2))
            bfuse_kernels.append("%s_%s_fused_bfuse" % (kname1, kname2))

    dim_decl_str = ""

    # Get hfuse kernel's dimension declarations
    for idx, kname in enumerate(hfuse_kernels):
        gridDim  = hfuseYAML[kname]["GridDim"]
        blockDim = hfuseYAML[kname]["BlockDim"]

        dim_decl_str += f"static unsigned int hfuse_gridDim_{idx}[] = {{{gridDim['X']}, {gridDim['Y']}, {gridDim['Z']}}};\n"
        dim_decl_str += f"static unsigned int hfuse_blockDim_{idx}[] = {{{blockDim['X']}, {blockDim['Y']}, {blockDim['Z']}}};\n"

    dim_decl_str += "//------------------------------------------------------------------------------------------\n"

    # Get bfuse kernel's dimension declarations
    for idx, kname in enumerate(bfuse_kernels):
        gridDim  = bfuseYAML[kname]["GridDim"]
        blockDim = bfuseYAML[kname]["BlockDim"]

        dim_decl_str += f"static unsigned int bfuse_gridDim_{idx}[] = {{{gridDim['X']}, {gridDim['Y']}, {gridDim['Z']}}};\n"
        dim_decl_str += f"static unsigned int bfuse_blockDim_{idx}[] = {{{blockDim['X']}, {blockDim['Y']}, {blockDim['Z']}}};\n"
    
    dim_decl_str += "//------------------------------------------------------------------------------------------\n"

    return dim_decl_str
#-----------------------------------------------------------------------------------------------
def get_fuse_func_decl_str(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    kernels       = []
    hfuse_kernels = []
    bfuse_kernels = []

    # Get fused kernel names
    for kname1 in fusion_sets[0]["Set"]:
        for kname2 in fusion_sets[1]["Set"]:
            kernels.append(kname1)
            kernels.append(kname2)
            hfuse_kernels.append("%s_%s_fused_hfuse" % (kname1, kname2))
            bfuse_kernels.append("%s_%s_fused_bfuse" % (kname1, kname2))

    func_decl_str = ""

    # Remove duplicated elements
    kernels       = list(set(kernels))
    hfuse_kernels = list(set(hfuse_kernels))
    bfuse_kernels = list(set(bfuse_kernels))

    # Get kernel's function declarations
    for kname in kernels:
        func_decl_str += f"extern \"C\" void {kname}(float* __restrict, float* __restrict, float* __restrict);\n"

    func_decl_str += "//------------------------------------------------------------------------------------------\n"

    # Get hfuse kernel's function declarations
    for kname in hfuse_kernels:
        func_decl_str += f"extern \"C\" void {kname}(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);\n"

    func_decl_str += "//------------------------------------------------------------------------------------------\n"

    # Get bfuse kernel's function declarations
    for kname in bfuse_kernels:
        func_decl_str += f"extern \"C\" void {kname}(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);\n"

    func_decl_str += "//------------------------------------------------------------------------------------------\n"

    return func_decl_str
#-----------------------------------------------------------------------------------------------
def get_kernel_macro_str(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    kernel_macro_str = ""

    # Get kernel1's macro
    kernel_macro_str += "#define ASSIGN_KERNEL1(I_shape, F_shape, O_shape, func, gridDim, blockDim, idx) \\\n"
    kernel_macro_str += "  do \\\n"
    kernel_macro_str += "  { \\\n"
    kernel_macro_str += "    switch (idx) \\\n"
    kernel_macro_str += "    { \\\n"
    for idx, kname in enumerate(fusion_sets[0]["Set"]):
        kernel_macro_str += f"    case {idx}: \\\n"
        kernel_macro_str += f"      I_shape  = kernel1_I_shape_{idx}; \\\n"
        kernel_macro_str += f"      F_shape  = kernel1_F_shape_{idx}; \\\n"
        kernel_macro_str += f"      O_shape  = kernel1_O_shape_{idx}; \\\n"
        kernel_macro_str += f"      func     = {kname}; \\\n"
        kernel_macro_str += f"      gridDim  = kernel1_gridDim_{idx}; \\\n"
        kernel_macro_str += f"      blockDim = kernel1_blockDim_{idx}; \\\n"
        kernel_macro_str += f"      break; \\\n"
    kernel_macro_str += "    } \\\n"
    kernel_macro_str += "  } while(0)\n"

    kernel_macro_str += "//------------------------------------------------------------------------------------------\n"

    # Get kernel2's macro
    kernel_macro_str += "#define ASSIGN_KERNEL2(I_shape, F_shape, O_shape, func, gridDim, blockDim, idx) \\\n"
    kernel_macro_str += "  do \\\n"
    kernel_macro_str += "  { \\\n"
    kernel_macro_str += "    switch (idx) \\\n"
    kernel_macro_str += "    { \\\n"
    for idx, kname in enumerate(fusion_sets[1]["Set"]):
        kernel_macro_str += f"    case {idx}: \\\n"
        kernel_macro_str += f"      I_shape  = kernel2_I_shape_{idx}; \\\n"
        kernel_macro_str += f"      F_shape  = kernel2_F_shape_{idx}; \\\n"
        kernel_macro_str += f"      O_shape  = kernel2_O_shape_{idx}; \\\n"
        kernel_macro_str += f"      func     = {kname}; \\\n"
        kernel_macro_str += f"      gridDim  = kernel2_gridDim_{idx}; \\\n"
        kernel_macro_str += f"      blockDim = kernel2_blockDim_{idx}; \\\n"
        kernel_macro_str += f"      break; \\\n"
    kernel_macro_str += "    } \\\n"
    kernel_macro_str += "  } while(0)\n"

    kernel_macro_str += "//------------------------------------------------------------------------------------------\n"

    return kernel_macro_str
#-----------------------------------------------------------------------------------------------
def get_fuse_macro_str(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        loggging.error("Number of fusion sets are only 2.")
        exit(0)

    fuse_macro_str = ""

    # Get hfuse kernel's macro
    fuse_macro_str += "#define ASSIGN_HFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shpae, O2_shape, func, gridDim, blockDim, idx1, idx2) \\\n"
    fuse_macro_str += "  do \\\n"
    fuse_macro_str += "  { \\\n"
    fuse_macro_str += "    switch (idx1) \\\n"
    fuse_macro_str += "    { \\\n"
    for idx1, kname1 in enumerate(fusion_sets[0]["Set"]):
        fuse_macro_str += f"    case {idx1}: \\\n"
        fuse_macro_str += "      switch (idx2) \\\n"
        fuse_macro_str += "      { \\\n"
        for idx2, kname2 in enumerate(fusion_sets[1]["Set"]):
            fuse_macro_str += f"      case {idx2}: \\\n"
            fuse_macro_str += f"        I1_shape = kernel1_I_shape_{idx1}; \\\n"
            fuse_macro_str += f"        F1_shape = kernel1_F_shape_{idx1}; \\\n"
            fuse_macro_str += f"        O1_shape = kernel1_O_shape_{idx1}; \\\n"
            fuse_macro_str += f"        I2_shape = kernel2_I_shape_{idx2}; \\\n"
            fuse_macro_str += f"        F2_shape = kernel2_F_shape_{idx2}; \\\n"
            fuse_macro_str += f"        O2_shape = kernel2_O_shape_{idx2}; \\\n"
            fuse_macro_str += f"        func     = {kname1}_{kname2}_fused_hfuse; \\\n"
            fuse_macro_str += f"        gridDim  = hfuse_gridDim_{idx1 * len(fusion_sets[1]['Set']) + idx2}; \\\n"
            fuse_macro_str += f"        blockDim = hfuse_blockDim_{idx1 * len(fusion_sets[1]['Set']) + idx2}; \\\n"
            fuse_macro_str += f"        break; \\\n"
        fuse_macro_str += "      } \\\n"
        fuse_macro_str += "      break; \\\n"
    fuse_macro_str += "    } \\\n"
    fuse_macro_str += "  } while(0)\n"

    fuse_macro_str += "//------------------------------------------------------------------------------------------\n"

    # Get bfuse kernel's macro
    fuse_macro_str += "#define ASSIGN_BFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func, gridDim, blockDim, idx1, idx2) \\\n"
    fuse_macro_str += "  do \\\n"
    fuse_macro_str += "  { \\\n"
    fuse_macro_str += "    switch (idx1) \\\n"
    fuse_macro_str += "    { \\\n"
    for idx1, kname1 in enumerate(fusion_sets[0]["Set"]):
        fuse_macro_str += f"    case {idx1}: \\\n"
        fuse_macro_str += "      switch (idx2) \\\n"
        fuse_macro_str += "      { \\\n"
        for idx2, kname2 in enumerate(fusion_sets[1]["Set"]):
            fuse_macro_str += f"      case {idx2}: \\\n"
            fuse_macro_str += f"        I1_shape = kernel1_I_shape_{idx1}; \\\n"
            fuse_macro_str += f"        F1_shape = kernel1_F_shape_{idx1}; \\\n"
            fuse_macro_str += f"        O1_shape = kernel1_O_shape_{idx1}; \\\n"
            fuse_macro_str += f"        I2_shape = kernel2_I_shape_{idx2}; \\\n"
            fuse_macro_str += f"        F2_shape = kernel2_F_shape_{idx2}; \\\n"
            fuse_macro_str += f"        O2_shape = kernel2_O_shape_{idx2}; \\\n"
            fuse_macro_str += f"        func     = {kname1}_{kname2}_fused_bfuse; \\\n"
            fuse_macro_str += f"        gridDim  = bfuse_gridDim_{idx1 * len(fusion_sets[1]['Set']) + idx2}; \\\n"
            fuse_macro_str += f"        blockDim = bfuse_blockDim_{idx1 * len(fusion_sets[1]['Set']) + idx2}; \\\n"
            fuse_macro_str += f"        break; \\\n"
        fuse_macro_str += "      } \\\n"
        fuse_macro_str += "      break; \\\n"
    fuse_macro_str += "    } \\\n"
    fuse_macro_str += "  } while(0)\n"

    fuse_macro_str += "//------------------------------------------------------------------------------------------\n"

    return fuse_macro_str
#-----------------------------------------------------------------------------------------------
def get_macro_h(infoYAML, kernelsYAML, hfuseYAML, bfuseYAML):
    # Get kernel's shape declarations
    shape_decl_str    = get_shape_decl_str(infoYAML)
    dim_decl_str      = get_dim_decl_str(infoYAML, kernelsYAML)
    fuse_dim_decl_str = get_fuse_dim_decl_str(infoYAML, hfuseYAML, bfuseYAML)
    func_decl_str     = get_fuse_func_decl_str(infoYAML)
    kernel_macro_str  = get_kernel_macro_str(infoYAML)
    fuse_macro_str    = get_fuse_macro_str(infoYAML)

    macro_h = shape_decl_str + dim_decl_str + fuse_dim_decl_str \
              + func_decl_str + kernel_macro_str + fuse_macro_str

    return macro_h
#-----------------------------------------------------------------------------------------------