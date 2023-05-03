# Fast Code for Image Interpolation
This is a part of submission for the course 18-646: How To Write Fast Code II

### Workspace
We conducted our experiments on the ECE cluster. The ECE machine is `ece020.ece.local.cmu.edu` . <br>
The path to our workspace directory is `/afs/ece.cmu.edu/usr/arexhari/Public/645-project` 

## Code Execution Instructions:
## GPU Code Execution
1. `make gpu-bl` - is to run all the implementaions of our GPU kernel. This command will all the 4 variants of our implementation across image sizes form 2x2 to 8192x8192. The four different variants are: naive implmentation, Pragma Unroll, Manual unroll and interleaved instructions

| File | Description ||
| (src/src/gpu-bi-linear-loop.cu)[Naive Implementation] | This is the naive implementation with for loops for all computations ||

## Baseline Codes
### Nearest Neighbors
1. `make bi-linear-run` is to demonstrate the functional correctness of our nearest neighbor interpolation algorithm
  - It makes use of a inputs/640-480.jpg and writes the output to results/640x480-nn.jpg
2. `make bi-linear-performance` is to demonstrate the performance of our code. After the code is run you can see the csv results at plots/kernelPerformanceNN.csv
3. `make parallel` is to demonstrate the parallelisation of our code. After the code is run you can see the csv results at plots/blParallel.csv

### Bi-Linear Kernel
1. `make nn-run` is to demonstrate the functional correctness of our nearest neighbor interpolation algorithm
  - It makes use of a inputs/640-480.jpg and writes the output to results/640x480-bl.jpg
2. `make nn-performance` is to demonstrate the performance of our code. After the code is run you can see the csv results at plots/kernelPerformance.csv
3. `make parallelNN` is to demonstrate the parallelisation of our code. After the code is run you can see the csv results at plots/nnParallel.csv

