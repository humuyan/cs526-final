import timeit
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import math

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import topi, autotvm
import logging
from datetime import datetime
import sys
# Enable debug logs
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

target_name = 'cuda -libs=cublas'

tuning_rounds = 64
assert len(sys.argv) == 6
bs, n_head, h, w, size_per_head = map(int, sys.argv[1:])

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = f'ansor.{time_now}.json'
# log_file = "ansor.2022-07-15.16-09-59.json"
print('log file:', log_file)

@auto_scheduler.register_workload
def div(bs, n_head, h, w, size_per_head, dtype):
    x = te.placeholder(shape=(bs, n_head, h * w, size_per_head + 1), name="x", dtype=dtype)
    out = te.compute(
        (bs, n_head * size_per_head, h, w),
        lambda i, j, k, l: x[i, j // size_per_head, w * k + l, j % size_per_head] / (x[i, j // size_per_head, w * k + l, 16] + 1e-15),
        name="out"
    )
    return [x, out]

################################################################################
args = (bs, n_head, h, w, size_per_head, dtype)
tasks = [
    tvm.auto_scheduler.SearchTask(
        func=div,
        args=args,
        target=target)
]

################################################################################
# Set Parameters for Auto-Scheduler
tuner = auto_scheduler.TaskScheduler(tasks, strategy='round-robin')
measure_ctx = auto_scheduler.LocalRPCMeasureContext(
    timeout=20, min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=tuning_rounds*len(tasks),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    runner=measure_ctx.runner,
    verbose=2,
)

tuner.tune(tune_option)
print("Tuning accomplished.")

funcs = []
for i, task in enumerate(tasks):
    sch, args = task.apply_best(log_file)
    funcs.append(tvm.build(sch, args, target))
    print('Lowering with GPU splits: \n ===================== \n{}'.format(tvm.lower(sch, args, simple_mode=True)))
    print(funcs[i].imported_modules[0].get_source())