# agentic-ai-training

You can find this repo at https://github.com/olcf/ai-training-series

The first step is to clone it and navigate to the training module:

```bash
git clone https://github.com/olcf/ai-training-series.git
cd agentic-ai-training
```

## Setting up your environment

For this training you will need the following:
* A Python environment that can install from the included `requirements.txt`
* A `sqlite3` installation

### Python environment

At OLCF, we use `miniforge3` on our production machines, which is available on GitHub here: https://github.com/conda-forge/miniforge

You can find install instructions here: https://github.com/conda-forge/miniforge#install

The basic instructions for Mac/Linux/WSL are:
1. Run `curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"`
2. Run `bash Miniforge3-$(uname)-$(uname -m).sh`

Once installed, you may need to open a new terminal window to activate `miniforge3`.
You can set up the environment easily as follows:

`conda create -p ./agentic-ai-training-env python=3.14`

You should press `y` when asked if you would like to install Python and necessary packages.

Next, activate the environment:

`conda activate ./agentic-ai-training-env`

Finally, you can install the packages needed for the module with:

`pip install -r requirements-python314.txt`

### `sqlite3` installation

On WSL, you should be able to use your package manager to install `sqlite3`.
If you have a default configuration running Ubuntu, you can run the following:

`sudo apt install sqlite3`

On MacOS, you can install with Homebrew:

`brew install sqlite3`


## Tutorial Flow

The project supports a single tutorial entrypoint through `main.py`, which can run one step at a time or the full RAG pipeline.

Before running the `query`/`chat` step, export your SambaNova API key:

`export SAMBANOVA_API_KEY=your_key_here`

The tutorial flow is:

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

You will be dropped into a chat prompt, which you can leave by typing `exit` and hitting enter.

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
