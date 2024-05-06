nvcc -gencode arch=compute_70,code=sm_70 -O3 ./timing.cu -o timing -lcublas -lcudnn -lcurand -std=c++14 -L/usr/local/cuda/lib -I/usr/local/cuda/include
./timing
