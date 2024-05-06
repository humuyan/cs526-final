# This script ingests an ONNX model and determines if the number of tasks TVM can extract
# TVM extract_tasks() perform op fusion where possible as part of traversal of the computation graph
# Lowered TIR schedueled are returned as tunable tasks\

import onnx
from onnx import shape_inference

import sys
import os
import argparse

from kernel_profiler import get_input_node_info

import tvm
from tvm.relay.frontend import from_onnx
from tvm import meta_schedule as ms

from utils import configure_target

import pandas as pd

from operator_fission import perform_op_fission
import taso

import os
import sys


def op_fission_taso(graph, model_name, USE_TASO=True, tmp_filepath = "tmp.onnx", export_filepath="../cases/cases_onnx/mobile_cases/"):
    print("Warning: sometimes TASO picks up non-fissoned graph; works if op_fission and taso are run separately one by one")
    fission_graph = perform_op_fission(graph)
    onnx.save(graph, tmp_filepath)
    
    if USE_TASO:
        model = taso.load_onnx(tmp_filepath)
        taso_graph = taso.optimize(model, print_subst=True)
        new_model = taso.export_onnx(taso_graph)
        new_model_si = shape_inference.infer_shapes(new_model)
        
        try:
            onnx.checker.check_model(new_model_si)
            print("The model is valid!")
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
            sys.exit()
        
        onnx.save(new_model_si, export_filepath + model_name)
    else:
        onnx.save(fission_graph, export_filepath + model_name)
        
    os.remove(tmp_filepath)


def estimate_kernels_generated(TARGET, onnx_path):
    
    onnx_model = onnx.load(onnx_path)

    # shape_dict, input_name, input_shape, DTYPE = get_input_node_info(onnx_model)

    # TVM ONNX to TensorIR parser
    mod, params = from_onnx(onnx_model, freeze_params=True)
    print("Loaded model\n")
    
    # extracted_tasks = extract_task_from_relay(mod, TARGET, params)
    print(f"Extracting tasks... for {onnx_path}\n")
    extracted_tasks = ms.relay_integration.extract_tasks(mod, target=TARGET, params=params)
    
    weighted_sum = 0
    for task in extracted_tasks:
        print(task.task_name, task.weight)
        weighted_sum = weighted_sum + task.weight
        
    return weighted_sum
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        type=str,
        default="candy",
        help="Model to profile",
    )
    
    parser.add_argument(
        "--target_device",
        type=str,
        default="cuda",
        help="select: cuda, adreno",
    )
    
    parser.add_argument(
        "--optimize",
        type=bool,
        default=False,
        help="select: true, false",
    )
    
    parser.add_argument(
        "--use_taso",
        type=bool,
        default=False,
        help="select: true, false",
    )
    
    MODEL = parser.parse_args().model
    TARGET_DEVICE = parser.parse_args().target_device
    OPTIMIZE = parser.parse_args().optimize
    USE_TASO = parser.parse_args().use_taso
    
    print(f"MODEL:{MODEL}\nTARGET_DEVICE:{TARGET_DEVICE}\nOPTIMIZE: {OPTIMIZE}\nUSE_TASO:{USE_TASO}\n")
    
    # MODELS = os.listdir('../cases/cases_onnx/mobile_cases/')
    EXPORT_FILEPATH = "../cases/cases_onnx/op_fiss_taso/"
    
    task_estimate = pd.DataFrame(columns=["Model Name", "# TVM Tasks"])
    
    # for MODEL in MODELS:
    if 1:
        if ".onnx" in MODEL:
            onnx_path = "../cases/cases_onnx/" + MODEL
            print(onnx_path)
            DEVICE, TARGET, RPC_CONFIG = configure_target(TARGET_DEVICE)
            
            if OPTIMIZE:
                try:
                    print("Optimize Op Fission + TASO")
                    
                    model = onnx.load(onnx_path)
                    op_fission_taso(model, model_name=MODEL, USE_TASO=USE_TASO, 
                                    export_filepath=EXPORT_FILEPATH)
                    del model
                    onnx_path = EXPORT_FILEPATH + MODEL
                except TypeError:
                    print(f"{MODEL} encountered TypeError")
                    sys.exit()
            

            num_tasks = estimate_kernels_generated(TARGET, onnx_path)
            task_estimate.loc[len(task_estimate)] = [MODEL, num_tasks]
    
    print(task_estimate)
