model="/mnt/nvme3n1/model_public/Atom7B/checkpoint-101k-32kl_9k-sft_19k-ppo_500-chat"

python convert.py  \
--outfile /tmp/atom_1b_old \
$model
