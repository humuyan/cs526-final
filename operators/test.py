from compute_bound_profiler import profile_conv, profile_gemm

n = 16
c = 128
h = 64
w = 64
f = 128
r = 3
flops = n * c  * h * w * f * r * r * 2
print(flops / profile_conv(n, c, h, w, f, r, 1, 1, 1, 1, 1, 0) * 1000 / 1024 / 1024 / 1024 / 1024, "TFLOPS")
b = 64
m = 1024
n = 1024
k = 1024
flops = b * m * n * k * 2
duration = profile_gemm(b, m, n, k, 0, 0, 1)
print(duration)
print(flops / duration * 1000 / 1024 / 1024 / 1024 / 1024, "TFLOPS")
