import sched
from google.protobuf.json_format import MessageToDict
import logging
import numpy as np
import onnx
import os

import re

import shutil

import tvm
from tvm import relay

from tvm.contrib import graph_executor
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
from tvm.meta_schedule.database.json_database import JSONDatabase
from tvm.relay.frontend import from_onnx
from tvm.runtime import Module, NDArray, vm
from tvm.support import describe
from tvm.target import Target

# from tvm import meta_schedule as ms
import tvm.auto_scheduler as auto_scheduler

from typing import Callable, Dict, List, Optional, Union

from datetime import datetime

NUM_RUNS = 20
NUM_REPEAT = 3
MIN_REPEAT_MS = 150
CPU_FLUSH = True

NUM_TRIALS_PER_ITER = 64
NUM_TRIALS = 1024
ADAPTIVE_TRAINING = True
BACKEND = "graph"

time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
LOG_FILE = f'ansor.{time_now}.json'

describe()

# %%
def get_input_node_info(onnx_model):
    # TVM from_onnx() requires shape_dict to be a dictionary of node name: List of dimensions
    shape_dict = {}
    input_name = ""
    DTYPE = ""
    input_shape = []
    
    for _input in onnx_model.graph.input:
        # ONNX format returns graph nodes as protobuf object
        m_dict = MessageToDict(_input)
        print("input_name : ", m_dict['name'])
        print("input_shape: ", m_dict["type"]["tensorType"]['shape'])
        print("input_dtype: ", m_dict["type"]["tensorType"]['elemType'])
        dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]
        input_shape = [int(d.get("dimValue")) for d in dim_info]
        input_name = m_dict["name"]
        shape_dict[input_name] = input_shape
        
        # TODO: Convert enum elemType to required datatype
        DTYPE = "float32" if m_dict["type"]["tensorType"]['elemType'] == 1 else "float32"
        
    return shape_dict, input_name, input_shape, DTYPE


def get_output(data, lib, dev, input_name):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, data)
        module.run()
        return module.get_output(0).numpy()


def profile_main(onnx_path="", onnx_model=None, WORK_DIR="./tune_kernels/", DB_WIPE=True):
    """_summary_

    Args:
        onnx_path (str): The path to the candidate cases stored as onnx files
        onnx_model (ModelProto, optional): Pass in an onnx model object directly is desired
        WORK_DIR (str, optional): location to store kernel tuning logs
        DB_WIPE (bool, optional): To disable wiping of database every iteration of kernel profiler. Defaults to True.
        TODO: The wiping of database is temporary until a better solution is found to reconcile task.task_name and Workload hash

    Returns:
        _type_: _description_
    """
    
    assert onnx_model is None and onnx_path != ""

    if onnx_model is None:
        onnx_model = onnx.load(onnx_path)

    shape_dict, input_name, input_shape, DTYPE = get_input_node_info(onnx_model)

    # TVM ONNX to TensorIR parser
    mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)

    # Reference: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # Page 22 Compute Capabilities table
    DEVICE = tvm.cuda()
    TARGET = tvm.target.cuda(arch='sm_70' , options="-max_threads_per_block=1024 -max_shared_memory_per_block=96000")

    hardware_params = auto_scheduler.HardwareParams(
        num_cores=-1,
        vector_unit_bytes=16,
        cache_line_bytes=64,
        max_shared_memory_per_block=96000,
        max_threads_per_block=1024,
        # The value `max_local_memory_per_block` is not used in AutoScheduler,
        # but is required by the API.
        max_local_memory_per_block=12345678,
        max_vthread_extent=8,
        warp_size=32,
    )
            
    data = tvm.nd.array(np.random.randn(*input_shape).astype(DTYPE), DEVICE)
    
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target=TARGET,
                                                        hardware_params=hardware_params,
                                                        include_simple_tasks=True)

    for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
        print(
            f"==== Task {idx}: {task.desc} "
            f"(weight {task_weight} key: {task.workload_key}) ====="
        )
        print(task.compute_dag)
        
    
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, strategy='round-robin')
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
                        timeout=20, min_repeat_ms=300
                    )
    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=NUM_TRIALS*len(tasks),
                        measure_callbacks=[auto_scheduler.RecordToFile(LOG_FILE)],
                        runner=measure_ctx.runner,
                        verbose=2,
                    )
    
    tuner.tune(tune_option)

    funcs = []
    pattern = re.compile(r'"thread_extent" = (\d+)')
    for i, task in enumerate(tasks):
        sch, args = task.apply_best(LOG_FILE)
        funcs.append(tvm.build(sch, args, TARGET))
        print(str(tvm.lower(sch, args, simple_mode=True)))
        print(pattern.findall(str(tvm.lower(sch, args, simple_mode=True))))
        print(funcs[i].imported_modules[0].get_source())


if __name__=="__main__":
    model_name = "relu"
    DB_WIPE = False

    onnx_path = "../cases/candidate_cases/" + model_name + ".onnx"
    profile_main(onnx_path=onnx_path, DB_WIPE=DB_WIPE)

