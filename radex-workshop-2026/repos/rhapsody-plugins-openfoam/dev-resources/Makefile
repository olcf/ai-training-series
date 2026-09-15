ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= python3

.PHONY: docs docs-serve

docs:
	cd $(ROOT)/.. && $(PYTHON) -m mkdocs build --clean

docs-serve:
	cd $(ROOT)/.. && $(PYTHON) -m mkdocs serve -a 0.0.0.0:8001
