import onnx_graphsurgeon as gs
import onnx
import numpy as np

input_onnx = "/home/muyan/yolox_bs1_res416.onnx"
output_onnx = "yolox_bs1_res416_replaced.onnx"

graph = gs.import_onnx(onnx.load(input_onnx))
new_nodes = []
slice_id = 255

def new_slice_node(input, starts: int, ends: int, axis: int):
    global slice_id
    slice_id += 1
    starts_i = gs.Constant(name=f"Slice_{slice_id}_starts", values=np.array([starts]))
    ends_i = gs.Constant(name=f"Slice_{slice_id}_ends", values=np.array([ends]))
    axes_i = gs.Constant(name=f"Slice_{slice_id}_axes", values=np.array([axis]))
    output = gs.Variable(name=f"Slice_{slice_id}_output", dtype=input.dtype)
    new_nodes.append(gs.Node(op="Slice", name=f"Slice_{slice_id}", inputs=[input, starts_i, ends_i, axes_i], outputs=[output]))
    return output

for node in graph.nodes:
    if node.op == "Split":
        axis = node.attrs["axis"]
        size0, size1 = node.inputs[1].values
        node.o(0, 0).inputs = [new_slice_node(node.inputs[0], 0, size0, axis)]
        node.o(0, 1).inputs = [new_slice_node(node.inputs[0], size0, size0 + size1, axis)]
    else:
        new_nodes.append(node)

graph.nodes = new_nodes
graph.cleanup(True, True, True)
onnx.save(gs.export_onnx(graph), output_onnx)