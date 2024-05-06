import onnxruntime as ort
import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--executable", type=str)
args = parser.parse_args()

model = gs.import_onnx(onnx.load(args.model))
assert len(model.inputs) == 1
assert len(model.outputs) == 1
# generate input
input_shape = model.inputs[0].shape
input_tensor = np.random.random(input_shape).astype(np.float32)
with open("input.txt", "w") as f:
    f.write(str(np.prod(input_shape)) + '\n')
    for i in input_tensor.flatten():
        f.write(str(i) + ' ')
# run our inference
os.system(f"{args.executable} input.txt output.txt")
# read output from file and save to a tensor
output_shape = model.outputs[0].shape
output_tensor = np.array([float(i) for i in open("output.txt", "r").read().split()]).reshape(output_shape)
# run inference with onnxruntime
session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
result = session.run([model.outputs[0].name], {model.inputs[0].name: input_tensor})
# compare output
error = np.abs(result[0] - output_tensor)
print(f"Maximium Error: {error.max()} (absolute), {error.max() / np.abs(result[0]).max()} (relative)")
print(f"Mean Error: {error.mean()} (absolute), {error.mean() / np.abs(result[0]).max()} (relative)")