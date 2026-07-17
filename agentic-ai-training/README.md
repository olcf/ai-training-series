# agentic-ai-training

## Prerequisites

For this training you will need the following:
* Access to a computer with an internet connection.
* (Optional) An OLCF account.

OLCF users are provided the following, learners outside of OLCF will need to follow additional instructions in each section:
* A Python environment with PyTorch, capable of installing the additional requirements in `requirements.txt`.
* An embedding model for embedding document text.
* A large language model

## Log in to Frontier/Odo

OLCF users can follow this training from Frontier/Odo.

To log into Frontier:
```bash
ssh <username>@frontier.olcf.ornl.gov
```

To log into Odo:
```bash
ssh <username>@odo.olcf.ornl.gov
```

## Shell environment

This section is for OLCF users. If you are not running on OLCF systems, you can skip to the Python environment section.

Load requisite modules and environment variables:
```bash
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Because using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
```

## Python environment

All learners will need to follow this section to create or use a Python environment.

At OLCF, we use `miniforge3` on our production machines, which is available on GitHub here: https://github.com/conda-forge/miniforge

You can use `minifroge3` on both Frontier and Odo by using the provided module, which is what we loaded in the previous step.

<details>
<summary>Instructions to set up your own environment</summary>

Adapted from the [PyTorch on Frontier User Docs](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html)

Create the base environment:
```bash
conda create -p ./agentic-ai-training-env python=3.14 -y
conda activate ./agentic-ai-trainig-env
```

Install PyTorch for AMD GPUs, which are what Frontier and Odo use:
```bash
# if you are not using AMD GPUs, or are using another version of ROCm, you will need to choose another --index-url
# If you are using NVIDIA GPUs, you should be able to skip this step
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/rocm7.1
```

Install remaining requirements:
```bash
pip install -r requirements-python314.txt
```

</details>

The environment is already provided for OLCF users. You can load it with the following commands:
Frontier:
```bash
conda activate /lustre/orion/stf007/world-shared/agentic-ai-training/agentic-ai-training-env
```

Odo:
```bash
conda activate /gpfs/wolf2/olcf/stf007/world-shared/agentic-ai-training/agentic-ai-training-env
```

## Setting up the ChromaDB

We can now initialize the ChromaDB we will be using for the RAG portion of this training.

With the included `config.py`, `create_contextual_chunks.py` and your new Python environment, you should be able to run the following:

```bash
python create_contextual_chunks.py
```

This script should generate a `chunks.jsonl` file which you can parse through to see the results of chunking the input
data.

## Generating Embeddings

<details>
<summary>Instructions to pull `nomic-embed-text-v2-moe`</summary>

Follow these instructions to gain access to `nomic-embed-text-v2-moe`.

You will need to set up git-lfs on Frontier/Odo.
```bash
module load git-lfs
git-lfs install
git clone https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
```

</details>

OLCF users can use pre-fetched models on Frontier and Odo.

The script defaults to using the `nomic-embed-text-v2-moe` model for embeddings, and the paths are programmed
into the script.

If you are running on Odo, your run command needs the `--odo` flag.

Frontier:
```bash
python generate_embeddings.py
```

Odo:
```bash
python generate_embeddings.py --odo
```

<details>
<summary>Specifying which model to run or run outside OLCF</summary>

Outside OLCF systems, you can choose to run with other models.
You can follow the above instructions for pulling `nomic-embed-text-v2-moe` to your local filesystem, and point to it:

```bash
python generate_embeddings.py --embedding-model local/path/to/nomic-embed-text-v2-moe
```

If you are following along on your own computer, you can run the following to automatically fetch the embedding model. 

```bash
python generate_embeddings.py --embedding-model nomic-ai/nomic-embed-text-v2-moe
```
</details>

`python create_contextual_chunks.py`


## Current Tutorial Flow

The project now supports a single tutorial entrypoint through `main.py`, which can run one step at a time or the full RAG pipeline.

Before running the query/chat step, export your SambaNova API key:

`export SAMBANOVA_API_KEY=your_key_here`

The current tutorial flow is:

1. Chunk the source documents into retrieval-ready text chunks
2. Embed those chunks and store them in ChromaDB
3. Retrieve relevant chunks and send them to the chat model for a grounded answer

In other words, the pipeline is:

`documents -> chunks -> embeddings -> retrieval -> grounded answer`

### Run Each Step

To run just the chunking step:

`python3 main.py --step chunk`

To run the embedding step:

`python3 main.py --step embed`

To run the retrieval/chat step with a question:

`python3 main.py --step query --query "Summarize the papers"`

To run the full pipeline end to end:

`python3 main.py --step all --query "What security risks are discussed?"`

### Step Summaries

`chunk`

Reads the source documents, cleans and splits them into overlapping chunks, and writes chunk records to `chunks.jsonl` or Chroma depending on the selected output mode.

`embed`

Reads the chunk records, loads the embedding model, converts each chunk into a vector, and stores the vectors in a Chroma collection.

`query`

Embeds the user question, retrieves the most relevant chunks from Chroma, and sends the grounded context to the SambaNova chat model for a cited answer.

`all`

Runs the full tutorial pipeline in order: chunk, embed, and then query.

### Verbose Mode

If you want to see a more detailed trace of the pipeline, add `--verbose` to any command:

`python3 main.py --step all --query "Summarize the papers" --verbose`

Verbose mode prints the functions being executed along with a short one-sentence summary of what each function is doing.


## Fine-Tuning Quick Start

To generate fine-tuning examples from the existing RAG corpus:

`python3 finetuning/data_creation/generate_examples.py`

This writes a generated dataset to:

`finetuning/generated_train.jsonl`

To fine-tune a tiny example model on that generated dataset:

`python3 finetuning/train_model.py`

This saves the fine-tuned model to:

`finetuning/model_output`
