.PHONY: clean-cpp
clean-cpp:
	rm -rf src/cpp/
	find src -name '*.cpp' -exec rm -f {} \;

.PHONY: clean
clean: clean-cpp

.PHONY: clobber
clobber: clean
	rm -rf ./build/ ./src/radex.egg-info/
	pip uninstall -y radex
