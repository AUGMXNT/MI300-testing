'''
mamba install protobuf
pip install accelerate
accelerate launch test-nemotron.py
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_path = 'mgoin/Nemotron-4-340B-Instruct-hf'
# model_path = 'nvidia/Minitron-4B-Base'
tokenizer  = AutoTokenizer.from_pretrained(model_path)

dtype  = torch.bfloat16
model  = AutoModelForCausalLM.from_pretrained(
             model_path, 
             torch_dtype=dtype, 
             device_map="auto"
         )

# Prepare the input text
prompt = 'Complete the paragraph: our solar system is'
inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

# Generate the output
outputs = model.generate(inputs, max_length=100)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)
