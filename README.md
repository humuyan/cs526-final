# CS526 Final Project

## TL; DR

We have uploaded the generated optimized code for candy in `test/forward_pass.cu`. Just run the following command to check the correctness of a 21M codegen result:

```bash
cd test
nvcc ./forward_pass.cu -O3 -lcudnn -gencode arch=compute_80,code=sm_80
python test.py --model test_cases/candy_bs1.onnx --executable ./a.out
```

## Environment Preparation

### Set up Python Environment

```bash
source activate pytorch
pip install nvidia-pyindex
pip install tornado psutil 'xgboost<1.6.0' cloudpickle onnx onnx-graphsurgeon transformers netron sortedcontainers pulp==2.7.0
```

### Install TVM

```bash
git clone --recursive https://github.com/balamurugan15/tvm-kernel-mapper.git tvm
export TVM_HOME=`realpath tvm`
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
cd tvm
mkdir build && cd build
cp ../cmake/config.cmake .
cmake ..
make -j4
python -c "import tvm; print(tvm.__version__)"
```

Showing `0.13.dev0` means that TVM has been installed correctly.

### Clone Korch Repo and Compile Korch's Compute-bound Profiler

```bash
git clone https://github.com/humuyan/cs526-final korch
cd korch/operators
./build.sh
cp compute_bound_profiler.cpython-39-x86_64-linux-gnu.so ../framework
```

## Test Cases

All the unit tests are under `test/test_cases` with ONNX format.

## Run Korch

### Operator Fission

```bash
cd ../framework
python operator_fission.py [input_onnx_path] [output_onnx_path]
```

You can use `netron` to visualize the ONNX graph to see the difference after operator fission.

### Kernel Orchestration

```bash
python calc.py [input_onnx_path] [gpu_type] --code_output_dir [code_output_dir] 
```

See `python calc.py -h` for detailed instructions.

Known issue: first time to run `calc.py` may lead to codegen failure. You may need to run `calc.py` twice to get the final generated code. Since we use a database to store the results of tuned subgraph, the second run will be much faster.

The kernel orchestration strategy and overall latencies will be shown in the terminal. If `code_output_dir` is specified, the generated code will be shown in `code_output_dir/subgraph0/forward_pass.cu`.

## Validate Correctness

First, compile `forward_pass.cu` into an executable:

```bash
nvcc nvcc ./forward_pass.cu -O3 -lcudnn -gencode arch=compute_[gpu_arch_number],code=sm_[gpu_arch_number]
```

Then run `test.py` in the `test` directory:

```bash
python test.py --model [input_onnx_path] --executable [executable_path]
```

The error numbers will be output to the terminal.
