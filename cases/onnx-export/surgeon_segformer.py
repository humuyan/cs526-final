import onnx_graphsurgeon as gs
import onnx

def surgeon(input_onnx, output_onnx, inputs, outputs):
	graph = gs.import_onnx(onnx.load(input_onnx))
	tensors = graph.tensors()
	graph.inputs = [tensors[input] for input in inputs]
	graph.outputs = [tensors[output] for output in outputs]
	graph.cleanup(True, True, True)
	onnx.save(gs.export_onnx(graph), output_onnx)

surgeon("segformer_bs16_res512_fission.onnx", "segformer_subgraph0_bs16.onnx", ["input.1"], ["/model/segformer/encoder/Transpose_output_0"])
surgeon("segformer_bs16_res512_fission.onnx", "segformer_subgraph1_bs16.onnx", ["/model/segformer/encoder/Transpose_output_0"], ["/model/segformer/encoder/Transpose_1_output_0"])
surgeon("segformer_bs16_res512_fission.onnx", "segformer_subgraph2_bs16.onnx", ["/model/segformer/encoder/Transpose_1_output_0"], ["/model/segformer/encoder/Transpose_2_output_0"])
surgeon("segformer_bs16_res512_fission.onnx", "segformer_subgraph3_bs16.onnx", ["/model/segformer/encoder/Transpose_2_output_0"], ["/model/segformer/encoder/Transpose_3_output_0"])
surgeon("segformer_bs16_res512_fission.onnx", "segformer_subgraph4_bs16.onnx", ["/model/segformer/encoder/Transpose_output_0"] + [f"/model/segformer/encoder/Transpose_{i}_output_0" for i in range(1, 4)], ["1558"])