# Contributing to RHAPSODY

Thank you for your interest in contributing to RHAPSODY! We welcome contributions from the community and are pleased to have you join us.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Pre-commit Hooks (Mandatory)](#pre-commit-hooks-mandatory)
- [Making Contributions](#making-contributions)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to [info@radical.org](mailto:info@radical.org).

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Areas for Contribution

We welcome contributions in several areas:

- **Bug fixes**: Help us identify and fix issues
- **Feature development**: Implement new functionality
- **Documentation**: Improve and expand our documentation
- **Testing**: Add test coverage and improve test quality
- **Performance**: Optimize existing code
- **Platform support**: Add support for new HPC platforms
- **Backend implementations**: Develop new execution backends

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rhapsody.git
cd rhapsody

# Add the upstream remote
git remote add upstream https://github.com/radical-cybertools/rhapsody.git
```

### 2. Set Up Development Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests to ensure everything is working
pytest tests/

# Check that imports work correctly
python -c "import rhapsody; print('RHAPSODY installed successfully')"
```

## Pre-commit Hooks (Mandatory)

**IMPORTANT**: Pre-commit hooks are mandatory for all contributions. They ensure code quality and consistency across the project.

### Installation

```bash
# Install pre-commit hooks (required for all contributors)
pre-commit install
```

### What the Hooks Do

Our pre-commit configuration includes:

- **Code Formatting**: Automatic formatting with `ruff format`
- **Linting**: Code quality checks with `ruff check`
- **Documentation**: Docstring formatting with `docformatter`
- **Security**: Secret detection and security checks
- **Spell Checking**: Typo detection with custom RHAPSODY dictionary
- **File Formatting**: Trailing whitespace, end-of-file fixes

### Running Pre-commit Hooks

```bash
# Run on all files (do this before your first commit)
pre-commit run --all-files

# Pre-commit will automatically run on each commit
git commit -m "Your commit message"

# If hooks fail, fix the issues and commit again
# The hooks will prevent commits with code quality issues
```

### Bypassing Hooks (Not Recommended)

```bash
# Only use in emergency situations
git commit --no-verify -m "Emergency commit message"
```

**Note**: Pull requests with bypassed pre-commit checks will be rejected.

## Making Contributions

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new feature branch
git checkout -b feature/your-feature-name
# OR for bug fixes:
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Follow our [Code Style Guidelines](#code-style-guidelines)
- Add tests for new functionality
- Update documentation as needed
- Ensure pre-commit hooks pass

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit (pre-commit hooks will run automatically)
git commit -m "Add feature: description of your changes"
```

### Commit Message Guidelines

Use clear, descriptive commit messages:

```bash
# Good examples:
git commit -m "Add support for SLURM GPU partitions"
git commit -m "Fix memory leak in concurrent backend"
git commit -m "Update documentation for platform abstraction"

# Include issue numbers when applicable:
git commit -m "Fix task state transition bug (fixes #123)"
```

## Code Style Guidelines

### Python Code Style

We use `ruff` for both linting and formatting. The pre-commit hooks will automatically enforce these rules:

- **Line length**: 88 characters (Black-compatible)
- **Import sorting**: Organized with `isort` compatibility
- **Docstrings**: Google-style docstrings required for public APIs
- **Type hints**: Required for new functions and methods
- **Variable naming**: snake_case for functions/variables, PascalCase for classes

### Example Code Style

```python
"""Module docstring describing the purpose."""

from typing import Dict, List, Optional
import asyncio

from rhapsody.base import BaseBackend


class ExampleBackend(BaseBackend):
    """Example backend implementation.

    This class demonstrates the expected code style for RHAPSODY
    contributions.

    Args:
        config: Configuration dictionary for the backend.
        max_workers: Maximum number of concurrent workers.
    """

    def __init__(self, config: Dict[str, str], max_workers: int = 4) -> None:
        """Initialize the example backend."""
        super().__init__(config)
        self._max_workers = max_workers

    async def submit_task(self, task: Dict[str, str]) -> str:
        """Submit a task for execution.

        Args:
            task: Task dictionary containing execution parameters.

        Returns:
            Task ID for tracking execution.

        Raises:
            ValueError: If task configuration is invalid.
        """
        if not task.get("executable"):
            raise ValueError("Task must specify an executable")

        return await self._execute_task(task)
```

## Testing Requirements

### Test Coverage

- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests
- **Minimum coverage**: Aim for >90% code coverage
- **Integration tests**: Required for backend implementations

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=rhapsody --cov-report=html

# Run specific test categories
pytest tests/test_backends/
pytest tests/test_integration/
pytest -k "asyncflow"

# Run performance tests
pytest tests/test_performance/ -v
```

### Test Organization

```
tests/
├── test_asyncflow_integration.py    # AsyncFlow compatibility tests
├── test_backend_functionality.py    # Backend implementation tests
├── test_realworld_integration.py   # End-to-end integration tests
├── test_performance/               # Performance benchmarks
├── test_platforms/                # Platform abstraction tests
└── fixtures/                      # Shared test fixtures
```

### Writing Tests

```python
import pytest
import asyncio
from rhapsody.backends import ConcurrentExecutionBackend


class TestConcurrentBackend:
    """Test suite for concurrent execution backend."""

    @pytest.fixture
    async def backend(self):
        """Create a test backend instance."""
        backend = ConcurrentExecutionBackend()
        yield backend
        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_task_submission(self, backend):
        """Test basic task submission functionality."""
        task = {
            "uid": "test_task",
            "executable": "echo",
            "arguments": ["Hello, World!"]
        }

        futures = await backend.submit_tasks([task])
        assert "test_task" in futures

        result = await futures["test_task"]
        assert result["state"] == "DONE"
```

## Documentation

### Documentation Requirements

- **API Documentation**: All public functions must have docstrings
- **README Updates**: Update if adding new features
- **Examples**: Include usage examples for new functionality
- **Platform Documentation**: Document new platform support

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide usage examples where appropriate
- Keep documentation concise but comprehensive

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### 1. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 2. Create a Pull Request

1. Go to the RHAPSODY repository on GitHub
2. Click "New pull request"
3. Select your feature branch
4. Fill out the pull request template

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Pre-commit hooks pass

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## Review Process

### What to Expect

1. **Automated Checks**: GitHub Actions will run tests and checks
2. **Pre-commit Verification**: Ensure all hooks passed
3. **Code Review**: Maintainers will review your code
4. **Feedback**: You may be asked to make changes
5. **Approval**: Once approved, your PR will be merged

### Review Criteria

- **Code Quality**: Follows style guidelines and best practices
- **Testing**: Adequate test coverage for changes
- **Documentation**: Appropriate documentation updates
- **Functionality**: Changes work as intended
- **Compatibility**: No breaking changes without discussion

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Branches

- `main`: Stable releases
- `dev`: Development branch
- `release/x.y.z`: Release preparation branches

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: [info@radical.org](mailto:info@radical.org) for direct contact

### Before Asking for Help

1. Check existing issues and discussions
2. Review the documentation
3. Try the development setup steps
4. Include relevant error messages and system information

## Issue Reporting

### Bug Reports

Use the bug report template and include:
- RHAPSODY version
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

Use the feature request template and include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## Recognition

Contributors will be:
- Listed in the project's AUTHORS file
- Mentioned in release notes for significant contributions
- Invited to join the contributor team for ongoing contributions

## License

By contributing to RHAPSODY, you agree that your contributions will be licensed under the same [MIT License](LICENSE.md) that covers the project.
