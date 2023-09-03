
model="/mnt/data_online/models/llama/llama-2-7b-chat"

# chinese 7B
model="/mnt/data/zhangzheng/data/llama2/sft_7b_chat/checkpoint-1200_merge"

# llama 13B模型
model="/mnt/data_online/models/llama/models--meta-llama--Llama-2-13b-chat-hf"

# llama 3B 测试
model="/mnt/data/wuyongyu/train/AtomGPT/train/pretrain_3B/llama_3b" 

# llama 1B  测试
model="/mnt/data/wuyongyu/train/AtomGPT/train/pretrain_3B/llama_1b"

# 新的 llama 1B 
model="/mnt/data/wuyongyu/train/AtomGPT/train/pretrain_3B/llama_1b"

# Atom 1B
model="/mnt/data/zhangzheng/data/Atom-1B/model_sft_atomgpt_14000_0828/checkpoint-8500_merge"

# Aom 7B
# model="/mnt/data/zhangzheng/data/AtoM-7B/model_sft_atomgpt_14000_0823/checkpoint-10000_merge/"
# /tmp/atom_7b

# model="/mnt/data_online/models/llama/models--meta-llama--Llama-2-7b-chat-hf"
# /tmp/llama_7b

python convert.py  \
--outfile /tmp/atom_1b_old \
$model
