import torch

a, b = [torch.randn(64, 1024, 1024) for _ in range(2)]

class BMM(torch.nn.Module):
	def forward(self, a, b):
		return torch.bmm(a, b)

torch.onnx.export(
	BMM(),
	(a, b),
	"bmm.onnx"
)