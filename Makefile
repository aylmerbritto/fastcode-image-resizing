CFLAGS = `pkg-config --cflags opencv` -mavx -mavx2 -mfma
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

kernel: src/kernelImage.cpp
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<


memory: src/memory.cpp
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<	

nn: src/nnImage.cpp
	@echo ==========================================
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

bm : src/benchmarkSpec.cpp
	@echo ==========================================
	@echo Compiling benchmark script
	@echo ==========================================
	mkdir -p build/
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<

clean : 
	rm -rf build/*
	rm -rf results/*
	rm -rf plots/*
