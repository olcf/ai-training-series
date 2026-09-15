ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DIRS := $(ROOT)/../example $(ROOT)/../tests

PYTEST_ARGS ?=
PYTHON ?= python3

.PHONY: format lint black black-check isort isort-check test docs docs-serve

format: black isort

lint: black-check isort-check
VERBOSE ?= 0
VERBOSE_FLAG := $(if $(VERBOSE),-v,)

black:
	black $(VERBOSE_FLAG) --config $(ROOT)/radex-config.toml $(DIRS)

black-check:
	black $(VERBOSE_FLAG) --config $(ROOT)/radex-config.toml --check $(DIRS)

isort:
	isort $(VERBOSE_FLAG) --settings-file $(ROOT)/radex-config.toml $(DIRS)

isort-check:
	isort $(VERBOSE_FLAG) --settings-file $(ROOT)/radex-config.toml --check $(DIRS)

test:
	dragon -s -- -m pytest $(PYTEST_ARGS) -c $(ROOT)/radex-config.toml $(ROOT)/../tests

docs:
	@command -v doxygen >/dev/null 2>&1 || \
		echo "warning: 'doxygen' not found on PATH; the C++ API reference will fail to build."
	cd $(ROOT)/.. && $(PYTHON) -m mkdocs build --clean

docs-serve:
	@command -v doxygen >/dev/null 2>&1 || \
		echo "warning: 'doxygen' not found on PATH; the C++ API reference will fail to build."
	cd $(ROOT)/.. && $(PYTHON) -m mkdocs serve -a 0.0.0.0:8001
