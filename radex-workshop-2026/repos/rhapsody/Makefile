.PHONY: help install lint format test test-all test-unit test-unit-no-dragon test-unit-dragon test-integration test-quick test-regular test-dragon-only test-ci check setup-pre-commit
.DEFAULT_GOAL := help

# Python executable - override with PYTHON variable if needed
PYTHON ?= python3

help:  ## Show this help message
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  PYTHON - Python executable path (default: /usr/bin/python3)"
	@echo ""
	@echo "Examples:"
	@echo "  make test-quick                  # Run only non-Dragon tests (fast)"
	@echo "  make test-unit-dragon            # Run Dragon unit tests with 'dragon pytest'"
	@echo "  make test-integration-dragon     # Run Dragon integration tests with 'dragon pytest'"
	@echo "  make test-dask                   # Run only Dask backend tests"
	@echo "  PYTHON=/path/to/python make test-constants  # Use custom Python"

install:  ## Create virtual environment and install dependencies
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .[dev]

setup-pre-commit:  ## Install pre-commit hooks
	. .venv/bin/activate && pre-commit install

lint:  ## Run linter with auto-fix
	. .venv/bin/activate && ruff check . --fix

format:  ## Format code with ruff
	. .venv/bin/activate && ruff format .

lint-check:  ## Check linting without auto-fix
	. .venv/bin/activate && ruff check .

format-check:  ## Check code formatting without changes
	. .venv/bin/activate && ruff format --check .

# Test targets
test:  ## Run all tests via tox
	. .venv/bin/activate && tox

test-all:  ## Run all tests (unit + integration) with proper backend separation
test-all: test-unit test-integration

test-unit:  ## Run all unit tests (Dragon with dragon pytest, others with pytest)
test-unit: test-unit-no-dragon test-unit-dragon

test-unit-no-dragon:  ## Run unit tests excluding Dragon (uses regular pytest, fast)
	@echo "Running non-Dragon unit tests with pytest..."
	@$(PYTHON) -m pytest tests/unit/ \
		--ignore=tests/unit/test_backend_execution_dragon.py \
		--ignore=tests/unit/test_backend_data_dragon.py \
		-xvs

test-unit-dragon:  ## Run Dragon unit tests (requires 'dragon pytest', may hang other tests)
	@echo "Running Dragon unit tests with dragon pytest..."
	@dragon $(PYTHON) -m pytest tests/unit/test_backend_execution_dragon.py tests/unit/test_backend_data_dragon.py tests/unit/telemetry/test_adapters_dragon.py -xvs

test-integration:  ## Run all integration tests (non-Dragon + Dragon)
test-integration: test-integration-no-dragon test-integration-dragon

test-integration-no-dragon:  ## Run integration tests excluding Dragon (uses regular pytest)
	@echo "Running non-Dragon integration tests with pytest..."
	@RHAPSODY_TEST_MODE=regular $(PYTHON) -m pytest tests/integration/ -xvs

test-integration-dragon:  ## Run Dragon integration tests (requires 'dragon pytest')
	@echo "Running Dragon integration tests with dragon pytest..."
	@RHAPSODY_TEST_MODE=dragon dragon $(PYTHON) -m pytest tests/integration/ -xvs

test-quick:  ## Quick test - only non-Dragon tests (unit + integration, recommended for development)
test-quick: test-unit-no-dragon test-integration-no-dragon

test-regular:  ## Run all regular tests (excludes Dragon tests, uses pytest)
	@echo "Running all regular tests (excluding Dragon)..."
	@$(PYTHON) -m pytest tests/unit/ tests/integration/ \
		--ignore=tests/unit/test_backend_execution_dragon.py \
		--ignore=tests/unit/test_backend_data_dragon.py \
		-xvs

test-dragon-only:  ## Run Dragon tests only (requires 'dragon' launcher)
	@echo "Running Dragon tests..."
	@if command -v dragon >/dev/null 2>&1; then \
		dragon $(PYTHON) -m pytest tests/unit/test_backend_execution_dragon.py tests/unit/test_backend_data_dragon.py -xvs; \
	else \
		echo "Dragon launcher not available. Install Dragon and run with 'dragon' command."; \
		exit 1; \
	fi

test-ci:  ## CI test - regular tests + Dragon tests when the dragon launcher is available
	@echo "Running CI tests..."
	@$(MAKE) test-regular
	@echo ""
	@if command -v dragon >/dev/null 2>&1; then \
		$(MAKE) test-dragon-only; \
	else \
		echo "Dragon launcher not available on this Python version, skipping Dragon tests."; \
	fi

test-dask:  ## Run Dask backend tests only
	@echo "Running Dask tests..."
	@$(PYTHON) -m pytest tests/unit/test_backend_execution_dask_parallel.py -xvs

test-rp:  ## Run RADICAL-Pilot backend tests only
	@echo "Running RADICAL-Pilot tests..."
	@$(PYTHON) -m pytest tests/unit/test_backend_execution_radical_pilot.py -xvs

test-concurrent:  ## Run Concurrent backend tests only
	@echo "Running Concurrent tests..."
	@$(PYTHON) -m pytest tests/unit/test_backend_base.py -xvs

test-constants:  ## Run constants tests only
	@echo "Running constants tests..."
	@$(PYTHON) -m pytest tests/unit/test_backend_constants.py -xvs

test-performance:  ## Run API performance benchmarks
	@echo "Running API performance benchmarks..."
	@$(PYTHON) -m pytest tests/performance/test_api_performance.py -xvs

check:  ## Run lint checks via tox
	. .venv/bin/activate && tox -e lint-check

build:  ## Build distribution package
	python3 -m build
	tar -xvf dist/rhapsody-0.1.0.tar.gz
