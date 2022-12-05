CFLAGS = `pkg-config --cflags opencv` -mavx -mavx2 -mfma -O3
LIBS = `pkg-config --libs opencv`

versiontest : src/version.cpp
	mkdir -p build/
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	@echo ==========================================
	@echo Execution results
	@echo ==========================================
	./build/versiontest

benchmark : src/benchmark.cpp
	@echo ==========================================
	@echo Compiling benchmark script
	@echo ==========================================
	mkdir -p build/
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	@echo ==========================================
	@echo Executing benchmark script
	@echo ==========================================
	mkdir -p results/benchmark/
	./build/benchmark > plots/benchmarkTime.csv
	python scripts/plotPerformance.py

kernel: src/fastKernel.cpp
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	objdump -d ./build/kernel > kernel.S
	./build/kernel



memory: src/memory.cpp
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<	

nn: src/nnImage.cpp
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

bm : src/benchmarkSpec.cpp
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

performance: src/performanceTest.cpp
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

rp: clean bm performance
	sh performanceDriver.sh > plots/kernelPerformance.csv

clean : 
	rm -rf build/*
	rm -rf results/*
	rm -rf plots/*
