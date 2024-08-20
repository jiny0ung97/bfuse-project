#!/usr/bin/python3

import ncu_report

my_context = ncu_report.load_report("0_0_0.ncu-rep")

print(my_context.num_ranges())