CFLAGS = `pkg-config --cflags opencv`
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

clean : 
	rm -rf build/*
	rm -rf results/*