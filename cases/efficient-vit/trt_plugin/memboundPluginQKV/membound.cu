#include <cassert>

#ifdef _WIN32
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
using int64_t = long long;
using uint64_t = unsigned long long;
#else
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
#endif

extern "C" __global__ void __launch_bounds__(64)
    bs1_stage3_v(float *__restrict__ v, float *__restrict__ raw) {
    v[((((int) blockIdx.x) * 64) + ((int) threadIdx.x))] =
        (((((((int) blockIdx.x) * 13) + ((int) threadIdx.x)) % 17) == 16)
             ? 1.000000e+00f
             : raw[(
                   ((((((int) blockIdx.x) / 4352) * 786432) +
                     ((((((int) blockIdx.x) * 13) + ((int) threadIdx.x)) % 17) *
                      16384)) +
                    ((((((int) blockIdx.x) % 4352) * 64) +
                      ((int) threadIdx.x)) /
                     17)) +
                   524288)]);
}

extern "C" __global__ void __launch_bounds__(64)
    bs1_stage3_q(float *__restrict__ q, float *__restrict__ raw) {
    q[((((int) blockIdx.x) * 64) + ((int) threadIdx.x))] =
        ((0.000000e+00f < raw[(((((((int) blockIdx.x) >> 12) * 786432) +
                                 ((((int) threadIdx.x) & 15) * 16384)) +
                                ((((int) blockIdx.x) & 4095) * 4)) +
                               (((int) threadIdx.x) >> 4))])
             ? raw[(((((((int) blockIdx.x) >> 12) * 786432) +
                      ((((int) threadIdx.x) & 15) * 16384)) +
                     ((((int) blockIdx.x) & 4095) * 4)) +
                    (((int) threadIdx.x) >> 4))]
             : 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(64)
    bs1_stage3_k(float *__restrict__ k, float *__restrict__ raw) {
    k[((((int) blockIdx.x) * 64) + ((int) threadIdx.x))] =
        ((0.000000e+00f < raw[(((((((int) blockIdx.x) >> 12) * 786432) +
                                 ((((int) blockIdx.x) & 4095) * 64)) +
                                ((int) threadIdx.x)) +
                               262144)])
             ? raw[(((((((int) blockIdx.x) >> 12) * 786432) +
                      ((((int) blockIdx.x) & 4095) * 64)) +
                     ((int) threadIdx.x)) +
                    262144)]
             : 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(64)
    bs1_stage4_v(float *__restrict__ v, float *__restrict__ raw) {
    v[((((int) blockIdx.x) * 64) + ((int) threadIdx.x))] =
        (((((((int) blockIdx.x) * 13) + ((int) threadIdx.x)) % 17) == 16)
             ? 1.000000e+00f
             : raw[(
                   ((((((int) blockIdx.x) / 1088) * 196608) +
                     ((((((int) blockIdx.x) * 13) + ((int) threadIdx.x)) % 17) *
                      4096)) +
                    ((((((int) blockIdx.x) % 1088) * 64) +
                      ((int) threadIdx.x)) /
                     17)) +
                   131072)]);
}

extern "C" __global__ void __launch_bounds__(64)
    bs1_stage4_q(float *__restrict__ q, float *__restrict__ raw) {
    q[((((int) blockIdx.x) * 64) + ((int) threadIdx.x))] =
        ((0.000000e+00f < raw[(((((((int) blockIdx.x) >> 10) * 196608) +
                                 ((((int) threadIdx.x) & 15) * 4096)) +
                                ((((int) blockIdx.x) & 1023) * 4)) +
                               (((int) threadIdx.x) >> 4))])
             ? raw[(((((((int) blockIdx.x) >> 10) * 196608) +
                      ((((int) threadIdx.x) & 15) * 4096)) +
                     ((((int) blockIdx.x) & 1023) * 4)) +
                    (((int) threadIdx.x) >> 4))]
             : 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(64)
    bs1_stage4_k(float *__restrict__ k, float *__restrict__ raw) {
    k[((((int) blockIdx.x) * 64) + ((int) threadIdx.x))] =
        ((0.000000e+00f < raw[(((((((int) blockIdx.x) >> 10) * 196608) +
                                 ((((int) blockIdx.x) & 1023) * 64)) +
                                ((int) threadIdx.x)) +
                               65536)])
             ? raw[(((((((int) blockIdx.x) >> 10) * 196608) +
                      ((((int) blockIdx.x) & 1023) * 64)) +
                     ((int) threadIdx.x)) +
                    65536)]
             : 0.000000e+00f);
}

void membound(
    int bs, int feat_len, int w, float *raw, float *q, float *k, float *v) {
    if (w == 96) {
        bs1_stage3_q<<<49152, 64>>>(q, raw);
        bs1_stage3_k<<<49152, 64>>>(k, raw);
        bs1_stage3_v<<<52224, 64>>>(v, raw);
    } else {
        bs1_stage4_q<<<24576, 64>>>(q, raw);
        bs1_stage4_k<<<24576, 64>>>(k, raw);
        bs1_stage4_v<<<26112, 64>>>(v, raw);
    }
}