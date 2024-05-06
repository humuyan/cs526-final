import tvm
from tvm import te
import tvm.testing
import re
import tvm.auto_scheduler as auto_scheduler
import logging
from datetime import datetime
import sys
# Enable debug logs
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

target_name = 'cuda -libs=cublas'

bs = int(sys.argv[1])
n_heads = int(sys.argv[2])
feat_len = int(sys.argv[3])
w = int(sys.argv[4])
tuning_rounds = int(sys.argv[5])

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = f'ansor.{time_now}.json'
print('log file:', log_file)

@auto_scheduler.register_workload
def linear_attn_membound(bs, n_heads, feat_len, w, dtype):
    assert feat_len % n_heads == 0
    head_size = feat_len // n_heads
    assert head_size % 3 == 0
    raw = te.placeholder(shape=(bs, feat_len, w, w), name="raw", dtype=dtype)
    q = te.compute(
        shape=(bs, n_heads, w * w, head_size // 3),
        fcompute=lambda i, j, k, l: tvm.tir.if_then_else(
            raw[i, j * head_size + l, k // w, k % w] > 0,
            raw[i, j * head_size + l, k // w, k % w],
            tvm.tir.const(0, dtype)
        ),
        name="q"
    )
    k = te.compute(
        #shape=(bs, n_heads, head_size // 3, w * w),
        shape=(bs, n_heads, w * w, head_size // 3),
        fcompute=lambda i, j, l, k: tvm.tir.if_then_else(
            raw[i, j * head_size + k + head_size // 3, l // w, l % w] > 0,
            raw[i, j * head_size + k + head_size // 3, l // w, l % w],
            tvm.tir.const(0, dtype)
        ),
        name="k"
    )
    v = te.compute(
        shape=(bs, n_heads, w * w, head_size // 3 + 1),
        #shape=(bs, n_heads, w * w, head_size // 3),
        fcompute=lambda i, j, k, l: tvm.tir.if_then_else(
            l == head_size // 3,
            tvm.tir.const(1, dtype),
            raw[i, j * head_size + l + head_size // 3 * 2, k // w, k % w],
        ),
        #fcompute=lambda i, j, k, l:
        #    raw[i, j * head_size + l + head_size // 3 * 2, k // w, k % w],
        name="v"
    )
    return [raw, q, k, v]

################################################################################
args = (bs, n_heads, feat_len, w, dtype)
tasks = [
    tvm.auto_scheduler.SearchTask(
        func=linear_attn_membound,
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
pattern = re.compile(r'"thread_extent" = (\d+)')
for i, task in enumerate(tasks):
    sch, args = task.apply_best(log_file)
    funcs.append(tvm.build(sch, args, target))
    print(pattern.findall(str(tvm.lower(sch, args, simple_mode=True))))
    print(funcs[i].imported_modules[0].get_source())
