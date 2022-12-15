# Fast Code for Image Interpolation
This is a part of submission for the course 18-645: How To Write Fast Code 

### Workspace
We conducted our experiments on the ECE cluster. The ECE machine is `ece004.ece.local.cmu.edu` . <br>
The path to our workspace directory is `/afs/ece.cmu.edu/usr/arexhari/Public/645-project` 

## Code Execution Instructions:
### Nearest Neighbors
1. `make bi-linear-run` is to demonstrate the functional correctness of our nearest neighbor interpolation algorithm
  - It makes use of a inputs/640-480.jpg and writes the output to results/640x480-nn.jpg
2. `make bi-linear-performance` is to demonstrate the performance of our code. After the code is run you can see the csv results at plots/kernelPerformanceNN.csv

### Bi-Linear Kernel
1. `make nn-run` is to demonstrate the functional correctness of our nearest neighbor interpolation algorithm
  - It makes use of a inputs/640-480.jpg and writes the output to results/640x480-bl.jpg
2. `make nn-performance` is to demonstrate the performance of our code. After the code is run you can see the csv results at plots/kernelPerformance.csv

