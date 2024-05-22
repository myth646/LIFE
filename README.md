# LIFE
## SST2
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name sst2 --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100  --strategy split --leader_epoch 5 --num_users 1000 --seed 1762505000
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name sst2 --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 1000 --manual_distribution --seed 1762505000
  </code></pre>
</details>


## COLA
<details>
  <summary>IID</summary>
  <pre><code>python LIFE_noiid.py --model_name_or_path roberta-base --task_name cola --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --group 9 9 9 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 100 --manual_distribution --seed 3829044447
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noiid.py --model_name_or_path roberta-base --task_name cola --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2  --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 100 --manual_distribution --seed 3829044447
  </code></pre>
</details>



## STSB
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name stsb --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group 9 9 9 --group_proportions 0.33 0.33 0.33 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name stsb --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group 9 9 9 --group_proportions 0.2 0.3 0.5 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution
  </code></pre>
</details>

## QNLI
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name qnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 1000 --seed 2512399976
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noiid.py --model_name_or_path roberta-base --task_name qnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 1000 --seed 2512399976 --manual_distribution
  </code></pre>
</details>


## MNLI
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name mnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.0027 --leader_epoch 5 --num_users 10000 --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --seed 3301259171
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noiid.py --model_name_or_path roberta-base --task_name mnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.0027 --leader_epoch 5 --num_users 10000 --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --seed 3301259171 --manual_distribution
  </code></pre>
</details>



## MRPC
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name mrpc --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --seed 1046058099
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name mrpc --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution --seed 1046058099
  </code></pre>
</details>



## RTE
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name rte --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --seed 1661086535
  </code></pre>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name rte --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution  --seed 1661086535
  </code></pre>
</details>

# InclusiveFL
For IID problem, change the filename to 'InclusiveFL_default.py'. 
For Non-IID problem, change the filename to 'InclusiveFL_default_noniid.py'.
For InclusiveFL* and and all experiments without momentum distillation. Remove the '--mom_grad' script.



















