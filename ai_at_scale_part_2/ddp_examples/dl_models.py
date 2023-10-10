import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

def download_gpt(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"tokenizers/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(f"models/{model_name}")

def load_gpt(model_name):
    tokenizer = AutoTokenizer.from_pretrained(f"tokenizers/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}")
    
def main(args):
    if "gpt" in args[0]:
        download_gpt(args[0])
        #load_gpt(args[0])

if __name__ == "__main__":
    main(sys.argv[1:])

