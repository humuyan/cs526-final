#%%
import taso
import onnx
from onnx import version_converter, helper, shape_inference
#%%
# "softmax_subgraph_fission",
cases = [
        # "candy_fission",
         "segformer_bs1_res512_fission",
        #  "efficientvit_opset11_bs1_res2048",
         "yolov4_bs1_res416",
         "yolox_bs1_res416",
         "mobilevit_image_classification_const_initializer_fission"]

# "softmax_subgraph_fission",
cases = [
        # "candy_fission",
        #  "segformer_bs1_res512_fission",
        #  "efficientvit_opset11_bs1_res2048",
         "yolov4_bs1_res416",
         "yolox_bs1_res416"]

model_name = cases[1]
if model_name in ['candy_fission', 'segformer_bs1_res512_fission', 'softmax_subgraph_fission']:
    onnx_graph = f"../cases/onnx-export/op_fission/{model_name}.onnx"
else:
    onnx_graph = f"../cases/cases_onnx/{model_name}.onnx"

old_model = taso.load_onnx(onnx_graph)
print("Finished loading")
#%%0.3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
taso_graph = taso.optimize(old_model, print_subst=True)

# #%%
new_model = taso.export_onnx(taso_graph)

taso_export_shape_inferred = shape_inference.infer_shapes(new_model)

try:
    onnx.checker.check_model(taso_export_shape_inferred)
    print("The model is valid!")
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s" % e)

# %%
onnx.save(taso_export_shape_inferred, f"../cases/taso_export/{model_name}.onnx")
