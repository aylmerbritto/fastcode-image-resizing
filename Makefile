CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

versiontest : src/test.cpp
	mkdir -p build/
	g++ $(CFLAGS) $(LIBS) -o build/$@ $<
	@echo ==========================================
	@echo Execution results
	@echo ==========================================
	./build/versiontest
