import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1B
model_name_path="/mnt/data/zhangzheng/data/Atom-1B/model_sft_atomgpt_14000_0828/checkpoint-8500_merge"

# 7B
# model_name_path="/mnt/data/zhangzheng/data/AtoM-7B/model_sft_atomgpt_14000_0823/checkpoint-10000_merge/"

tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: 介绍一下北京\n</s><s>Assistant:'], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')

# [    1, 12968, 29901, 29871, 32400, 32306, 32164,    13,     2,     1, 4007, 22137, 29901]
print("input_ids:", input_ids)

id1 = [12968, 29901, 29871]
id1 = [29871, 1, 29950, 7889, 29901, 29871, 32400, 32306, 32164, 29905, 29876, 2, 1, 7900, 22137, 29901]
id1 = [259, 1, 12968, 29901, 29871, 32400, 32306, 32164, 29905, 29876, 2, 1, 7900, 22137, 29901]
id1 = [1, 12968, 29901, 29871, 32400, 32306, 32164, 29905, 29876, 2, 1, 4007, 22137, 29901]
# id1 = [1, 12968, 29901, 29871, 32400, 32306, 32164, 29966, 29900, 29916, 29900, 29909, 29958, 2, 1, 4007, 22137, 29901]
a = tokenizer.convert_ids_to_tokens(id1)
print("new:", a)

idd2 = [1, 12968, 29901, 29871, 32400, 32306, 32164,    13,     2,     1, 4007, 22137, 29901]
a = tokenizer.convert_ids_to_tokens(idd2)
print(a)
exit(0)


model = AutoModelForCausalLM.from_pretrained(model_name_path, 
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             load_in_8bit=True)
model =model.eval()
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
