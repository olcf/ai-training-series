from dataclasses import dataclass, field
from transformers import BertTokenizer
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

parser = HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
print(model_args, training_args)
trunc = truncate(model_args.max_len)
'''
max_data_rows = 14329267
max_data_shards = 14330
num_ranks = torch.distributed.get_world_size()
max_row = num_ranks*int(np.floor(max_data_shards/num_ranks)) - 1
data_len = int((max_row+1)*1000/num_ranks)
print('total data len per gpu:',data_len)
#dataset = wd.Dataset('/gpfs/alpine/world-shared/med106/g8o/webdata_full/part-{000000..%06d}.tar' % max_row,
#                     length=data_len, shuffle=True).decode('torch').rename(input_ids='pth').map_dict(input_ids=trunc).shuffle(1000)
'''
dataset = load_dataset("oscar", "unshuffled_deduplicated_af", cache_dir="oscar")

tokenizer = BertTokenizer.from_pretrained('/gpfs/alpine/world-shared/med106/g8o/pubmed_bert-vocab.txt')

train_dataset = dataset["train"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


context_length = 1024
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

config = BigBirdConfig(
    vocab_size=30_000,
    hidden_size=768,
    intermediate_size=3072,
    max_position_embeddings=1024,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2,
)

if model_args.model_name_or_path is not None:
    model = BigBirdForMaskedLM.from_pretrained(model_args.model_name_or_path)
else:
    model = BigBirdForMaskedLM(config=config)

trainer = MyTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

trainer.train()
trainer.save_model(model_args.savepath)

