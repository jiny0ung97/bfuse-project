#!/usr/bin/python3

import os, sys
import argparse
import logging
import subprocess
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set loggging environment
    logging.basicConfig(format="%(levelname)s (%(filename)s:%(lineno)s): %(message)s",
                        level=logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("example", action="store", help="example file path to generate test-suite")
    parser.add_argument("-o", action="store", default="test-suite", dest="file",
                        help="output file name of generated test-suite")

    # Get arguments
    args        = parser.parse_args()
    exam_path   = args.example
    output_path = args.file
    cur_path    = os.path.dirname(os.path.realpath(__file__))

    # Configure test-suite
    # ./configure_kernels.py -o ${test_suite_name} ../test-utils/tvm-kernels/examples/${test_example_name}
    print("[1/6] ========================= Configure test-suite =========================")
    configure_kernels_path = os.path.join(cur_path, "configure_kernels.py")

    command = [configure_kernels_path, "-o", output_path, exam_path]
    try:
        print("RUN: " + " ".join(command))
        # result = subprocess.run(command,
        #                         # stdout=subprocess.PIPE,
        #                         text=True,
        #                         check=True,
        #                         # timeout=10,
        #                         )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while configuring \'{output_path}\'.")
        exit(1)

    # Generate fused kernels (hfuse, bfuse)
    # ../horizontal-fuser/build/bin/horizontal-fuser -y ${test_suite_name}/config -o ${test_suite_name}/cuda ${test_suite_name}/config/fusions.yaml ${test_suite_name}/config/kernels.yaml cuda
    # ../horizontal-fuser/build/bin/horizontal-fuser -b -y ${test_suite_name}/config -o ${test_suite_name}/cuda ${test_suite_name}/config/fusions.yaml ${test_suite_name}/config/kernels.yaml ${test_suite_name}/cuda
    print("[2/6] ========================= Generate fused kernels (hfuse, bfuse) =========================")
    horizontal_fuser_path = os.path.join(cur_path, "../../horizontal-fuser/build/bin/horizontal-fuser")
    fuser_config_path     = os.path.join(output_path, "config")
    fuser_output_path     = os.path.join(output_path, "cuda")
    fuser_fusions_path    = os.path.join(fuser_config_path, "fusions.yaml")
    fuser_kernels_path    = os.path.join(fuser_config_path, "kernels.yaml")

    command1 = [horizontal_fuser_path, "-y", fuser_config_path, "-o", fuser_output_path, fuser_fusions_path, fuser_kernels_path, fuser_output_path]
    command2 = [horizontal_fuser_path, "-b", "-y", fuser_config_path, "-o", fuser_output_path, fuser_fusions_path, fuser_kernels_path, fuser_output_path]
    try:
        print("RUN: " + " ".join(command1))
        result1 = subprocess.run(command1,
                                # stdout=subprocess.PIPE,
                                text=True,
                                check=True,
                                # timeout=10,
                                )
        print("RUN: " + " ".join(command2))
        result2 = subprocess.run(command2,
                                # stdout=subprocess.PIPE,
                                text=True,
                                check=True,
                                # timeout=10,
                                )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while generating fused kernels.")
        exit(1)

    # Generate benchmark for test-suite
    # ./gen_test_suite.py ${test_suite_name}
    print("[3/6] ========================= Generate benchmark for test-suite =========================")
    gen_test_suite_path = os.path.join(cur_path, "gen_test_suite.py")

    command = [gen_test_suite_path, output_path]
    try:
        print("RUN: " + " ".join(command))
        result = subprocess.run(command,
                                # stdout=subprocess.PIPE,
                                text=True,
                                check=True,
                                # timeout=10,
                                )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while generating benchmark for test-suite.")
        exit(1)

    # Make benchmark
    # cd ${test_suite_name} && make && cd ..
    print("[4/6] ========================= Make benchmark =========================")
    tmp_path = os.getcwd()
    command1 = ["sed", "-i", "s/arch=compute_70,code=sm_70/arch=compute_80,code=sm_80/g", "Makefile"]
    command2 = ["make"]
    try:
        os.chdir(output_path)
        print("RUN: " + " ".join(command1))
        result1 = subprocess.run(command1,
                                 # stdout=subprocess.PIPE,
                                 text=True,
                                 check=True,
                                 # timeout=10,
                                 )
        print("RUN: " + " ".join(command2))
        result2 = subprocess.run(command2,
                                 # stdout=subprocess.PIPE,
                                 text=True,
                                 check=True,
                                 # timeout=10,
                                 )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while make.")
        exit(1)
    else :
        os.chdir(tmp_path)
    
    # Profile the benchmark
    # ./profile_test_suite.py -vme ${test_suite_name}
    print("[5/6] ========================= Profile the benchmark =========================")
    profile_test_suite_path = os.path.join(cur_path, "profile_test_suite.py")

    command = [profile_test_suite_path, "-vme", output_path]
    try:
        print("RUN: " + " ".join(command))
        result = subprocess.run(command,
                                # stdout=subprocess.PIPE,
                                text=True,
                                check=True,
                                # timeout=10,
                                )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while profiling the benchmark.")
        exit(1)

    # Draw profile results
    # ./draw_figure.py -o ${test_suite_name} ${test_suite_name}
    print("[6/6] ========================= Draw profile results =========================")
    draw_figure_path = os.path.join(cur_path, "draw_figure.py")
    result_path      = os.path.join(output_path, "profile")

    command = [draw_figure_path, "-o", result_path, output_path]
    try:
        print("RUN: " + " ".join(command))
        result = subprocess.run(command,
                                # stdout=subprocess.PIPE,
                                text=True,
                                check=True,
                                # timeout=10,
                                )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurs while drawing profile results.")
        exit(1)
#-----------------------------------------------------------------------------------------------