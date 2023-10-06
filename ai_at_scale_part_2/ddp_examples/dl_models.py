from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", cache_dir="tokenizers")
tokenizer.save_pretrained("./gptneoxtokenizer")
tokenizer =  AutoTokenizer.from_pretrained("./gptneoxtokenizer")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", cache_dir="gptneox")
model.save_pretrained("./gptneox22b")

model = AutoModelForCausalLM.from_pretrained("./gptneox22b")
print(model)

'''
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
tokenizer.save_pretrained("./gptxltokenizer")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
model.save_pretrained("./gpt2xl")
'''
