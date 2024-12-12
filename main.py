from sys import argv
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
np.random.seed(42)
torch.manual_seed(42)

def load_model_and_tokenizer(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path)

def generate(
        model, tok, text,
        do_sample=True, max_length=300, repetition_penalty=2.0,
        top_k=1, top_p=0.9, temperature=0.6,
        num_beams=3,
        no_repeat_ngram_size=3
):
    input_ids = tok.encode(text, return_tensors="pt")
    print(model.generate.__globals__['__file__'])
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
    )
    return list(map(tok.decode, out))


tok, model = load_model_and_tokenizer("sberbank-ai/rugpt3large_based_on_gpt2")

#«вспугнуть» и «спиной»

text = "Нужно быть предельно осторожным. Зверь позади. Сначала надо тихо подобраться к логову, чтобы вдруг его не"

generated = generate(model, tok, text, num_beams=1)
print(generated[0])
