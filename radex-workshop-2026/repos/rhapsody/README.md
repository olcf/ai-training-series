# RHAPSODY

[![Build Status](https://github.com/radical-cybertools/rhapsody/actions/workflows/ci.yml/badge.svg)](https://github.com/radical-cybertools/rhapsody/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://radical-cybertools.github.io/rhapsody/)
[![Python Version](https://img.shields.io/pypi/pyversions/rhapsody-py.svg)](https://pypi.org/project/rhapsody-py/)
[![PyPI Version](https://img.shields.io/pypi/v/rhapsody-py.svg)](https://pypi.org/project/rhapsody-py/)
[![License](https://img.shields.io/pypi/l/rhapsody-py.svg)](https://github.com/radical-cybertools/rhapsody/blob/main/LICENSE.md)

**RHAPSODY** – **R**untime for **H**eterogeneous **AP**plications, **S**ervice **O**rchestration and **DY**namism

A unified runtime for executing **AI and HPC workloads** on supercomputing infrastructures. RHAPSODY seamlessly integrates traditional scientific computing with AI inference, enabling complex workflows that combine simulation, analysis, and machine learning.

## What RHAPSODY Offers

- **Unified AI-HPC API**: Single interface for compute tasks and AI inference
- **Multi-Backend Execution**: Run on local machines, HPC clusters ([Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html)), or distributed systems ([Dask](https://docs.dask.org/en/stable/))
- **Async-First Design**: Native asyncio integration for efficient task orchestration
- **Integratable Design**: RHAPSODY is designed to be integratable with existing workflows and tools such as [AsyncFlow](https://github.com/radical-cybertools/radical.asyncflow) and [LangGraph/FlowGentic](https://github.com/stride-research/flowgentic).
- **Scale-Ready**: Scale your workload and workflows to thousands of tasks and nodes.

## Quick Example: AI-HPC Workflow

```python
import asyncio
from rhapsody.api import Session, ComputeTask, AITask
from rhapsody.backends import DragonExecutionBackend, DragonVllmInferenceBackend
from rhapsody.backends.ai.config import ModelConfig

async def main():
    # Initialize backends
    hpc_backend = await DragonExecutionBackend(name="hpc")
    ai_backend = await DragonVllmInferenceBackend(
        name="vllm", model=ModelConfig(model_name="Qwen2.5-7B", hf_token="...", tp_size=1)
    )

    # Create session with multiple backends
    async with Session(backends=[hpc_backend, ai_backend]) as session:

        # HPC simulation task
        simulation = ComputeTask(
            executable="./simulate",
            arguments=["--config", "params.yaml"],
            backend=hpc_backend.name
        )

        # AI analysis task
        analysis = AITask(
            prompt="Analyze the simulation results and identify key patterns...",
            backend=ai_backend.name
        )

        # Submit and execute
        await session.submit_tasks([simulation, analysis])

        # Wait for completion (tasks are awaitable!)
        sim_result = await simulation
        ai_result = await analysis

        print(f"Simulation: {sim_result['stdout']}")
        print(f"AI Analysis: {ai_result['response']}")

asyncio.run(main())
```

## Installation

```bash
# Basic installation
pip install rhapsody-py

# With specific backends
pip install rhapsody-py[dask]           # Dask distributed computing
pip install rhapsody-py[dragon]         # Dragon runtime (Python 3.10-3.12)

# Development
pip install rhapsody-py[dev]
```


## Documentation

- **Full Documentation**: https://radical-cybertools.github.io/rhapsody/
- **API Reference**: https://radical-cybertools.github.io/rhapsody/api/
- **Examples**: See `examples/` directory

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`make test-regular`)
6. Run code quality checks (`pre-commit run --all-files`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Reporting Issues

Please use the [GitHub issue tracker](https://github.com/radical-cybertools/rhapsody/issues) to report bugs or request features.

## License

RHAPSODY is licensed under the [MIT License](LICENSE.md).

## Acknowledgments

RHAPSODY is a collaborative project between two teams:

- **[RADICAL Research Group](https://radical.rutgers.edu/)** — Rutgers University, New Jersey: core architecture, API design, session management, telemetry, and backend abstraction.
- **[Hewlett Packard Enterprise (HPE)](https://www.hpe.com/)** — USA / Canada: DragonHPC runtime integration and Dragon backend development.

See the full [Team](https://radical-cybertools.github.io/rhapsody/project/team/) page for details.

### Related Projects

- [AsyncFlow](https://github.com/radical-cybertools/asyncflow): Asynchronous workflow management

## NSF-Funded Project

RHAPSODY is supported by the National Science Foundation (NSF) under Award ID [2103986](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103986). This collaborative project aims to advance the state-of-the-art in heterogeneous workflow execution for scientific computing.

### Citations

If you use RHAPSODY in your research, please cite:

```bibtex
@software{rhapsody2024,
  title={RHAPSODY: Runtime for Heterogeneous Applications, Service Orchestration and Dynamism},
  author={RADICAL Research Group, Rutgers University and Hewlett Packard Enterprise},
  year={2024},
  url={https://github.com/radical-cybertools/rhapsody},
  version={0.1.0}
}
```

## Support

- **Documentation**: https://radical-cybertools.github.io/rhapsody/
- **Issues**: https://github.com/radical-cybertools/rhapsody/issues
- **Discussions**: https://github.com/radical-cybertools/rhapsody/discussions
