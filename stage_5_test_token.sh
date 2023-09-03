text="<s>Human: 介绍一下北京\n</s><s>Assistant:"
# text="<unk><pad>Human: 介绍一下北京\n</s><s>Assistant:"
# text="怎么去北京"

/mnt/data/wuyongyu/train/llama.cpp/main -m \
/tmp/atom_1b_old \
-p "${text}"  \
--logdir ./logtxt 
