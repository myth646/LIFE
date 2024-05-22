# LIFE
# SST2


# COLA


# STSB
python LIFE.py --model_name_or_path roberta-base --task_name stsb --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group 9 9 9 --group_proportions 0.33 0.33 0.33 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 

python LIFE_noiid.py --model_name_or_path roberta-base --task_name stsb --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group 9 9 9 --group_proportions 0.2 0.3 0.5 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution

# QNLI

# MNLI


# MRPC


# RTE













