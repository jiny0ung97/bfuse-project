#!/usr/bin/python3

import os, sys
import argparse
import logging
import yaml

file_module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test-utils/test-suite-files")
sys.path.append(file_module_path)

import macro_h
import main_cc
import Makefile
import operation_h, operation_cu
import utils_h, utils_cc
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", action="store", help="benchmark path of generated test-suite")

    # Get arguments
    args   = parser.parse_args()
    output = args.file
    
    if not os.path.exists(output):
        logging.error("Given config path \"%s\" doesn't exist." % output)
        exit(1)

    # Parse YAML files
    config_path  = os.path.join(output, "config")
    info_path    = os.path.join(config_path, "info.yaml")
    kernels_path = os.path.join(config_path, "kernels.yaml")
    hfuse_path   = os.path.join(config_path, "hfuse_kernels.yaml")
    bfuse_path   = os.path.join(config_path, "bfuse_kernels.yaml")

    with open(info_path) as f:
        yaml_info = yaml.safe_load(f)
    with open(kernels_path) as f:
        yaml_kernels = yaml.safe_load(f)
    with open(hfuse_path) as f:
        yaml_hfuse = yaml.safe_load(f)
    with open(bfuse_path) as f:
        yaml_bfuse = yaml.safe_load(f)

    # Generate benchmark codes
    macro_h_path      = os.path.join(output, "macro.h")
    main_cc_path      = os.path.join(output, "main.cc")
    Makefile_path     = os.path.join(output, "Makefile")
    operation_cu_path = os.path.join(output, "operation.cu")
    operation_h_path  = os.path.join(output, "operation.h")
    utils_cc_path     = os.path.join(output, "utils.cc")
    utils_h_path      = os.path.join(output, "utils.h")
    
    with open(macro_h_path, "w+") as f:
        f.write(macro_h.get_macro_h(yaml_info, yaml_kernels, yaml_hfuse, yaml_bfuse))
    with open(main_cc_path, "w+") as f:
        f.write(main_cc.get_main_cc(yaml_info))
    with open(Makefile_path, "w+") as f:
        f.write(Makefile.get_Makefile_src())
    with open(operation_cu_path, "w+") as f:
        f.write(operation_cu.get_operation_cu(yaml_info))
    with open(operation_h_path, "w+") as f:
        f.write(operation_h.get_operation_h())
    with open(utils_cc_path, "w+") as f:
        f.write(utils_cc.get_utils_cc())
    with open(utils_h_path, "w+") as f:
        f.write(utils_h.get_utils_h())
#-----------------------------------------------------------------------------------------------