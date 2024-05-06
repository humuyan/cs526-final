import onnx
import onnx_graphsurgeon as gs

actual_batch_dim = 16

def apply(infile, outfile):
    model = onnx.load(infile)
    graph = gs.import_onnx(model)
    graph.inputs[0].shape[0] = actual_batch_dim
    for node in graph.nodes:
        node.outputs[0].shape[0] = actual_batch_dim
    model = gs.export_onnx(graph)
    onnx.save(model, outfile)

apply("/home/muyan/trt_logs/models/segformer_bs1_res512_fission.onnx", "segformer_bs16_res512_fission.onnx")