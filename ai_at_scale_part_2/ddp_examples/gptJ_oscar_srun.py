from dataclasses import dataclass, field
from transformers import BertTokenizer, AutoTokenizer
#from tokenizers.processors import BigBirdProcessing
from datasets import load_dataset
from transformers import BigBirdConfig
from transformers import BigBirdForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers import HfArgumentParser, set_seed
import os
import webdataset as wd
import numpy as np
import torch
from custom_trainer import MyTrainer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from typing import Any, Dict, Optional, Tuple

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None
    )
    savepath: str = field(
        default="./pretrained"
    )
    max_len: int = field(
        default=128
    )

class truncate(object):
    def __init__(self,max_len):
        self.max_len = max_len
    def __call__(self,doc):
        return doc[:self.max_len]

@dataclass
class MyTrainingArguments(TrainingArguments):
    master_address: Optional[str] = field(
        default=None,
        metadata={
            "help": "IP address of the first node."
        },
    )

master_addr = os.environ["MASTER_ADDR"]
print("Master address from main:", master_addr)

import os
from datetime import timedelta
master_port = "29500"

def setup_distributed_env(init_method=None, rank = 0, world_size=16): 
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = rank = comm.Get_rank()
    backend = None
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = "0"
    torch.distributed.init_process_group(backend,
                                        init_method=init_method,
                                        rank=rank,
                                        world_size=world_size)
    using_mpi = torch.distributed.get_backend() == 'mpi'

setup_distributed_env()


parser = HfArgumentParser((ModelArguments, MyTrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
print(model_args, training_args)
trunc = truncate(model_args.max_len)

dataset = load_dataset("oscar", "unshuffled_deduplicated_af", cache_dir="oscar")

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizers/EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("models/EleutherAI/gpt-j-6b")
#tokenizer = AutoTokenizer.from_pretrained("tokenizers/gpt2-xl")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

train_dataset = dataset["train"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

context_length = 128
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_dataset = train_dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names
)

#model = AutoModelForCausalLM.from_pretrained("models/gpt2-xl")
#model.half()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

trainer.train()
trainer.save_model(model_args.savepath)

