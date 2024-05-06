import onnx
import onnx_graphsurgeon as gs

src_path = "/home/muyan/efficientvit_old_opt_bs1_res2048.onnx"
target_path = "/home/muyan/trt_logs/models/efficientvit_old_opt_bs1_res2048.onnx"

print(f"saving to {target_path}")
graph = gs.import_onnx(onnx.load(src_path))
tot = 0


def surgeon(raw_name, qkv_name, kv_name):
	global tot
	for node in graph.nodes:
		if node.name == raw_name:
			raw = node.outputs[0]
		elif node.name == qkv_name:
			matmul_qkv = node
		elif node.name == kv_name:
			matmul_kv = node
	dtype = raw.dtype

	output_q = gs.Variable(name=f"output_q{tot}", dtype=dtype)
	output_k = gs.Variable(name=f"output_k{tot}", dtype=dtype)
	output_v = gs.Variable(name=f"output_v{tot}", dtype=dtype)
	membound_qkv = gs.Node(op="MemboundPluginQKV", inputs=[raw], outputs=[output_q, output_k, output_v])
	matmul_qkv.inputs[0] = output_q
	matmul_kv.inputs = [output_k, output_v]

	graph.nodes.append(membound_qkv)
	tot += 1


surgeon("/backbone/backbone.3/op_list.1/main/qkv/conv/Conv", "/backbone/backbone.3/op_list.1/main/MatMul_1", "/backbone/backbone.3/op_list.1/main/MatMul")
surgeon("/backbone/backbone.3/op_list.3/main/qkv/conv/Conv", "/backbone/backbone.3/op_list.3/main/MatMul_1", "/backbone/backbone.3/op_list.3/main/MatMul")
surgeon("/backbone/backbone.3/op_list.5/main/qkv/conv/Conv", "/backbone/backbone.3/op_list.5/main/MatMul_1", "/backbone/backbone.3/op_list.5/main/MatMul")
surgeon("/backbone/backbone.3/op_list.7/main/qkv/conv/Conv", "/backbone/backbone.3/op_list.7/main/MatMul_1", "/backbone/backbone.3/op_list.7/main/MatMul")
surgeon("/backbone/backbone.3/op_list.9/main/qkv/conv/Conv", "/backbone/backbone.3/op_list.9/main/MatMul_1", "/backbone/backbone.3/op_list.9/main/MatMul")
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), target_path)
