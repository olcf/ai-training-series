## RAG-Assisted Data Creation

This folder contains a small tutorial workflow for creating fine-tuning data
with the existing RAG pipeline.

The idea is:

1. start with a small set of seed topics or question styles
2. retrieve relevant paper chunks from Chroma
3. ask the SambaNova model to generate a training example grounded in that
   retrieved context
4. save the result as chat-format JSONL for fine-tuning

This keeps the dataset in the same domain as the RAG tutorial while showing
how retrieval can help produce higher-quality training data.

### Files

- `seed_topics.jsonl`
  Small set of seed prompts that define what kind of example to generate.
- `generate_examples.py`
  Uses retrieval plus SambaNova to create chat-format examples.

### Example Usage

Make sure the RAG pipeline has already been run so Chroma contains embeddings,
and export your SambaNova API key first:

`export SAMBANOVA_API_KEY=your_key_here`

Then run:

`python3 finetuning/data_creation/generate_examples.py`

This will create:

`finetuning/generated_train.jsonl`

You can control dataset size with:

`python3 finetuning/data_creation/generate_examples.py --examples-per-seed 5`
