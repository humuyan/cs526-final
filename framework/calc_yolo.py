import onnx_graphsurgeon as gs
import onnx
import sys
from queue import Queue
import compute_bound_profiler
from tqdm import tqdm
from kernel_profiler import profile_main

def visited_nodes(graph: gs.Graph) -> int:
    visited_nodes = []
    for tensor in graph.inputs:
        for node in tensor.outputs:
            visited_nodes.append(node)
    i = 0
    while i < len(visited_nodes):
        for node in visited_nodes[i].outputs[0].outputs:
            if node not in visited_nodes:
                visited_nodes.append(node)
        i += 1
    return i

graph = gs.import_onnx(onnx.load(sys.argv[1]))
q = Queue()
q.put(graph)
convs = []
membound_kernels = []
while not q.empty():
    cur : gs.Graph = q.get()
    conv = None
    for node in cur.nodes:
        if node.op == "Conv":
            conv = node
            break
    if conv:
        # identify conv
        kernel_size = conv.attrs["kernel_shape"]
        assert kernel_size[0] == kernel_size[1]
        stride = conv.attrs["strides"]
        assert stride[0] == stride[1]
        pad = conv.attrs["pads"]
        assert pad[0] == pad[1] == pad[2] == pad[3]
        dilation = conv.attrs["dilations"]
        assert dilation[0] == dilation[1]
        params = {"batch_size": conv.inputs[0].shape[0], "in_channels": conv.inputs[0].shape[1], "in_height": conv.inputs[0].shape[2], "in_width": conv.inputs[0].shape[3], "out_channels": conv.inputs[1].shape[0], "kernel_size": kernel_size[0], "stride": stride[0], "padding": pad[0], "dilation": dilation[0], "groups": conv.attrs["group"]}
        convs.append(params)
        # split the graph and check connectivity
        conv.outputs[0].outputs = []
        if visited_nodes(cur) != len(cur.nodes) - 1:
            cur_split = cur.copy()
            cur.outputs = [conv.inputs[0]]
            cur_split.inputs = [conv.outputs[0]]
            cur.cleanup(True, True, True)
            q.put(cur)
            cur_split.cleanup(True, True, True)
            q.put(cur_split)
        else:
            cur.inputs.append(conv.outputs[0])
            cur.outputs.append(conv.inputs[0])
            q.put(cur)
    else:
        membound_kernels.append(cur)

latency = 0
for kernel in tqdm(membound_kernels):
    t = sum([float(x[0]) for x in profile_main(onnx_model=gs.export_onnx(kernel), DB_WIPE=False, USE_GE_BENCHMARK=False).values()])
    latency += t
print(latency)
exit()

print(f"Convs: {len(convs)}, Membounds: {len(membound_kernels)}")
for kernel in membound_kernels:
    if len(kernel.nodes) != 2:
        print(kernel)
        print("--------------------------------------------")