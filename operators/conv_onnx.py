import torch

torch.onnx.export(
	torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
	torch.randn(1, 128, 224, 224),
	"conv.onnx"
)