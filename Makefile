CFLAGS = `pkg-config --cflags opencv` -mavx -mavx2 -mfma -O3
LIBS = `pkg-config --libs opencv`
COMPILER=/usr/local/cuda-11.4/bin/nvcc

GPUFLAGS = `pkg-config --cflags opencv` 

versiontest : src/version.cpp
	mkdir -p build/
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	@echo ==========================================
	@echo Running version test
	@echo ==========================================
	./build/versiontest

bmbl : src/benchmarkBL.cpp
	@echo ==========================================
	@echo Compiling the benchmark script for Bi-Linear
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -fopenmp -o build/$@ $<

bmnn : src/benchmarkBL.cpp
	@echo ==========================================
	@echo Compiling the benchmark script for Nearest-neighbors
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -fopenmp -o build/$@ $<

kernel: src/blKernel.cpp
	@echo ==========================================
	@echo Compiling the bi-linear kernel
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	objdump -d ./build/kernel > asm/kernelBL.S
	./build/kernel

nn: src/nnKernel.cpp
	@echo ==========================================
	@echo Compiling the Nearest Neighbor kernel
	@echo ==========================================
	mkdir -p asm
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	objdump -d ./build/nn > asm/kernelNN.S
	./build/nn

parallel: src/blParallel.cpp
	@echo ==========================================
	@echo Compiling the parallel script for Bi-Linear
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -fopenmp -o build/$@ $<
	@# objdump -d ./build/parallel > asm/parallelBL.S
	@# ./build/parallel > plots/blParallel.csv

parallelNN: src/nnParallel.cpp
	@echo ==========================================
	@echo Compiling the parallel script for Nearest Neighbors
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -fopenmp -o build/$@ $<
	objdump -d ./build/parallelNN > asm/parallelNN.S
	./build/parallelNN	> plots/nnParallel.csv

performance: src/blPerformance.cpp
	@echo ==========================================
	@echo Compiling the parallel script for Bi-Linear Script
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

performanceNN: src/nnPerformance.cpp
	@echo ==========================================
	@echo Compiling the parallel script for Nearest Neighbors
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

bi-linear-performance: clean bmbl performance
	sh performanceDriverBL.sh > plots/kernelPerformance.csv

nn-performance: clean bmnn performanceNN
	sh performanceDriverNN.sh > plots/kernelPerformanceNN.csv

bi-linear-run: clean kernel

nn-run: clean nn



gpu-bl-loop: src/gpu-bi-linear-loop.cu
	#NAIVE GPU CODE
	#=================
	@$(COMPILER) --resource-usage  $< -o build/$@ $(GPUFLAGS) $(LIBS) -Xcompiler -fopenmp -maxrregcount=255
	sh performanceDriverGPUBL.sh

gpu-bl-unroll: src/gpu-bi-linear-loop-pragma.cu
	#GPU PRAGMA UNROLL
	#=================
	@$(COMPILER) --resource-usage  $< -o build/$@ $(GPUFLAGS) $(LIBS) -Xcompiler -fopenmp -maxrregcount=255
	sh performanceDriverGPUBL-loop.sh

gpu-bl-manual-32: src/gpu-bi-linear-manual-unroll32.cu
	#GPU MANUAL UNROLL 32
	#=================
	@$(COMPILER) --resource-usage  $< -o build/$@ $(GPUFLAGS) $(LIBS) -Xcompiler -fopenmp -maxrregcount=255
	sh performanceDriverGPUBL-manual-32.sh

gpu-bl-interleaved: src/gpu-bi-linear-unrolled-interleave.cu
	#GPU INSTRUCTIONS INTERLEAVED
	#=================
	@$(COMPILER) --resource-usage  $< -o build/$@ $(GPUFLAGS) $(LIBS) -Xcompiler -fopenmp -maxrregcount=255
	sh performanceDriverGPUBL-il.sh

gpu-bl: clean gpu-bl-loop gpu-bl-unroll gpu-bl-manual-32 gpu-bl-interleaved 



clean : 
	rm -rf build/*
	rm -rf results/*
	rm -rf plots/*
