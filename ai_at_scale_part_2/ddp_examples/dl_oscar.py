from datasets import load_dataset
from transformers import BertTokenizer
dataset = load_dataset("oscar", "unshuffled_deduplicated_af", cache_dir="oscar")

