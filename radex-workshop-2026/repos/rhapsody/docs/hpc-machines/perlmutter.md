# NERSC Perlmutter

[Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/) is a Cray EX supercomputer at the National Energy Research Scientific Computing Center (NERSC), DOE. It features both GPU and CPU partitions, with GPU nodes equipped with 4x NVIDIA A100 40GB GPUs each. It runs a Cray PE software stack with Cray MPICH and Cray libfabric for high-speed interconnect.

This guide covers running RHAPSODY's AI inference workloads using the `DragonVllmInferenceBackend` on Perlmutter's GPU partition.

## Environment Setup

### 1. Create and activate a conda environment

```bash
conda create -n rhap_conda_env python=3.12
export PYTHONNOUSERSITE=1
conda activate rhap_conda_env
```

!!! note
    `PYTHONNOUSERSITE=1` prevents user-site packages from leaking into the conda environment, which is important on shared systems like Perlmutter.

### 2. Install RHAPSODY with Dragon AI (vLLM) support

```bash
pip install "rhapsody-py[ai]"
```

This installs the vLLM package thus, no separate install step needed.

### 3. Set the HuggingFace cache directory

Perlmutter's `$SCRATCH` filesystem is the recommended location for large model caches:

```bash
export HF_HOME=$SCRATCH/cache/huggingface
```

!!! note
    Add this to your `~/.bashrc` or job script so it persists across sessions. See the [NERSC docs](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/) for more on managing storage.

### 4. Configure Dragon's OFI runtime library

Dragon needs to be pointed to the Cray libfabric library for inter-node communication:

```bash
dragon-config add --ofi-build-lib=/opt/cray/libfabric/1.22.0/lib64
dragon-config add --ofi-include=/opt/cray/libfabric/1.22.0/include
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
```

!!! warning
    This step is required on Perlmutter. Without it, Dragon will fail to initialize its transport layer across nodes.

### 5. Download the model

`DragonVllmInferenceBackend` no longer reads a YAML config file — model settings are passed directly as a `ModelConfig` object (see the [Integrations guide](../integrations.md#dragon-vllm-inference-backend)). Perlmutter's `$HOME`/`$PSCRATCH` filesystems don't support the `flock()` calls the HuggingFace Hub client uses while downloading, so download to local disk first and copy the snapshot over rather than downloading directly to scratch:

```bash
# Download to node-local /tmp on the login node (flock works there)
export HF_HOME=/tmp/${USER}_hf_dl
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir /tmp/${USER}_hf_dl/Qwen2.5-0.5B-Instruct

# Move it to scratch — a plain file copy, no flock involved
mkdir -p $SCRATCH/models
rsync -av /tmp/${USER}_hf_dl/Qwen2.5-0.5B-Instruct/ $SCRATCH/models/Qwen2.5-0.5B-Instruct/
```

Then set, before launching your job, so the model loads from the local snapshot with no further Hub/lock calls:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Pass the local snapshot directory as `ModelConfig(model_name=...)` in your script (see step 6 below).

### 6. Allocate a GPU node

```bash
salloc -N 1 -t 30 -C gpu --gpus-per-node=1 --account=<your_account>
```

Once the allocation is granted, launch with the `dragon` command:

```bash
dragon <script.py>
```

---

## Example

### Simulation via Inference + Model Fine-Tuning

This example demonstrates a two-phase HPC-AI workflow on Perlmutter:

1. **Simulation phase** — run a batch of scientific simulations using LLM inference (`DragonVllmInferenceBackend`). Each `AITask` sends a domain-specific prompt and collects the generated output as synthetic simulation data.
2. **Training phase** — fine-tune a small model on the collected simulation outputs using a `ComputeTask` dispatched through `DragonExecutionBackend`.

Both phases run within the same RHAPSODY `Session`, sharing the same Dragon runtime across GPU and CPU resources.

```python title="ai-hpc.py"
import asyncio
import logging
import multiprocessing as mp
import os

import rhapsody
from rhapsody.api import AITask, ComputeTask, Session
from rhapsody.backends import DragonExecutionBackend, DragonVllmInferenceBackend
from rhapsody.backends.ai.config import HardwareConfig, ModelConfig

rhapsody.enable_logging(level=logging.INFO)

# Scientific simulation prompts — each represents one simulation scenario
SIMULATION_PROMPTS = [
    "Simulate the energy minimization of a water molecule using DFT. Report bond lengths and angles.",
    "Describe the outcome of a molecular dynamics simulation of a lipid bilayer at 310K over 100ns.",
    "Run a Monte Carlo simulation of a 2D Ising model at the critical temperature. Report magnetization.",
    "Simulate heat diffusion in a 1D rod with fixed boundary conditions. Report steady-state profile.",
    "Describe the result of a lattice Boltzmann simulation of fluid flow through a porous medium.",
    "Simulate protein folding of a 50-residue helix using coarse-grained MD. Report RMSD over time.",
    "Run a finite element simulation of stress distribution in a titanium beam under axial load.",
    "Simulate a 2D incompressible Navier-Stokes flow around a cylinder at Re=100. Report drag coefficient.",
]


def fine_tune(simulation_outputs: list):
    """Fine-tune a small causal LM on simulation-generated data."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading tokenizer and model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    dataset = Dataset.from_dict({"text": simulation_outputs})

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"], truncation=True, max_length=256, padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="./sim-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        fp16=True,
        save_strategy="no",
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model("./sim-finetuned")
    tokenizer.save_pretrained("./sim-finetuned")
    print("Fine-tuning complete. Model saved to ./sim-finetuned", flush=True)
    return "./sim-finetuned"


async def main():
    mp.set_start_method("dragon")

    execution_backend = await DragonExecutionBackend()

    inference_backend = await DragonVllmInferenceBackend(
        model=ModelConfig(
            model_name=f"{os.environ['SCRATCH']}/models/Qwen2.5-0.5B-Instruct",
            hf_token="",
            tp_size=1,
        ),
        hardware=HardwareConfig(num_nodes=1, num_gpus=1, node_offset=0),
        port=8001,
    )

    session = Session([execution_backend, inference_backend])

    async with session:
        # --- Phase 1: Simulation via inference ---
        print(f"--- Phase 1: Running {len(SIMULATION_PROMPTS)} simulations ---", flush=True)

        sim_tasks = [
            AITask(prompt=prompt, backend=inference_backend.name)
            for prompt in SIMULATION_PROMPTS
        ]

        futures = await session.submit_tasks(sim_tasks)
        await asyncio.gather(*futures)

        simulation_outputs = [t.response for t in sim_tasks if t.response]
        print(f"Collected {len(simulation_outputs)} simulation outputs", flush=True)

        for i, out in enumerate(simulation_outputs):
            print(f"  Sim {i}: {out[:80]}...", flush=True)

        # --- Phase 2: Fine-tune a small model on the simulation data ---
        print("--- Phase 2: Fine-tuning model on simulation outputs ---", flush=True)

        train_task = ComputeTask(
            function=fine_tune,
            args=[simulation_outputs],
            backend=execution_backend.name,
        )

        train_futures = await session.submit_tasks([train_task])
        await asyncio.gather(*train_futures)

        print(f"Training result: {train_task.stdout.strip()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

Run with:

```bash
dragon ai-hpc.py
```

!!! success "Expected Output"
    ```text
    --- Phase 1: Running 8 simulations ---
    Collected 8 simulation outputs
      Sim 0: The DFT energy minimization of a water molecule yields an O-H bond length of 0.96 Å...
      Sim 1: The lipid bilayer simulation at 310K shows stable membrane structure with an area per...
      Sim 2: At the critical temperature (T_c ≈ 2.269), the magnetization fluctuates around zero...
      Sim 3: The steady-state heat profile follows a linear gradient between the two fixed boundar...
      Sim 4: The porous medium simulation shows channeling effects with an effective permeability...
      Sim 5: The coarse-grained MD simulation shows the helix stabilizes around 2ns with RMSD < 2...
      Sim 6: Stress concentrations appear at the fixed end with a maximum von Mises stress of 320...
      Sim 7: The drag coefficient Cd ≈ 1.38, consistent with experimental values at Re=100. Karman...
    --- Phase 2: Fine-tuning model on simulation outputs ---
    Loading tokenizer and model: Qwen/Qwen2.5-0.5B-Instruct
    {'loss': 2.4231, 'epoch': 1.0}
    {'loss': 1.8902, 'epoch': 2.0}
    {'loss': 1.5214, 'epoch': 3.0}
    Fine-tuning complete. Model saved to ./sim-finetuned
    Training result: ./sim-finetuned
    ```

!!! note "Workflow pattern"
    This pattern — **inference-driven simulation followed by fine-tuning** — is a common AI-for-science loop: the LLM generates domain-specific synthetic data, which is then used to specialize a smaller model for downstream tasks such as surrogate modeling or experiment recommendation.

!!! tip "Scaling up"
    - Increase `SIMULATION_PROMPTS` to hundreds or thousands of scenarios to generate a larger training corpus
    - Use `HardwareConfig(num_gpus=4)` and `ModelConfig(tp_size=4)` on Perlmutter's A100 nodes for larger inference models
    - Replace `Qwen/Qwen2.5-0.5B-Instruct` in `fine_tune()` with any HuggingFace model that fits in GPU memory

---

## Contributors

<div class="grid cards" markdown>

-   :fontawesome-solid-user-tie: **Stephen Hudson**

    ---

    Argonne National Laboratory

    [:fontawesome-solid-arrow-up-right-from-square: Profile](https://www.anl.gov/profile/stephen-hudson)

-   :fontawesome-solid-user-tie: **Corneel Casert**

    ---

    National Energy Research Scientific Computing Center (NERSC)

    [:fontawesome-solid-arrow-up-right-from-square: Profile](https://www.nersc.gov/profile/corneel-casert)

</div>
