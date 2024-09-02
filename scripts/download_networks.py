#!/usr/bin/python3

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch, onnx

access_token = "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 "

# meta-llama/Meta-Llama-3-8B
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=access_token)
# model     = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# openai-community/gpt2
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model     = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# google-bert/bert-base-uncased
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")