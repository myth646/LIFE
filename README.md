# LIFE
- For InclusiveFL,when experimenting with IId problems, please change the filename to 'InclusiveFL_default.py'. And to 'InclusiveFL_default_noniid.py' when experimenting with Non-IId problems
- For InclusiveFL* and all experiments without momentum distillation, please remove the '--mom_grad' script.
## SST2
<details>
  <summary>IID </summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name sst2 --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100  --strategy split --leader_epoch 5 --num_users 1000 --seed 1762505000
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_sst2_accuracy(9_9_9).csv">IID(1:1:1) result</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_sst2_accuracy(19_6_1).csv">IID(19:6:1) result</a></li>
  </ul>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name sst2 --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 1000 --manual_distribution --seed 1762505000
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_sst2_accuracy_noniid.csv">Non-IID result</a></li>
  </ul>
</details>


## COLA
<details>
  <summary>IID </summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name cola --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group_proportions 0.33 0.33 0.33 --group 9 9 9 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 100 --seed 3829044447
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_cola_matthews_correlation(9_9_9).csv">IID(1:1:1) result</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_cola_matthews_correlation(19_6_1).csv">IID(19:6:1) result</a></li>
  </ul>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noiid.py --model_name_or_path roberta-base --task_name cola --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2  --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 100 --manual_distribution --seed 3829044447
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_cola_matthews_correlation_noniid.csv">Non-IID result</a></li>
  </ul>
</details>



## STSB
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name stsb --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group 9 9 9 --group_proportions 0.33 0.33 0.33 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100
  </code></pre>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_stsb_pearson(9_9_9).csv">IID(1:1:1) result (Pearson)</a></li>
    <li><a href="./output_diff_datasets/LIFE_stsb_spearmanr(9_9_9).csv">IID(1:1:1) result (Spearman)</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_stsb_pearson(19_6_1).csv">IID(19:6:1) result (Pearson)</a></li>
    <li><a href="./output_diff_datasets/LIFE_stsb_spearmanr(19_6_1).csv">IID(19:6:1) result (Spearman)</a></li>
  </ul>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name stsb --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --group 9 9 9 --group_proportions 0.2 0.3 0.5 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_stsb_pearson_noniid.csv">Non-IID result (Pearson)</a></li>
    <li><a href="./output_diff_datasets/LIFE_stsb_spearmanr_noniid.csv">Non-IID result (Spearman)</a></li>
  </ul>
</details>

## QNLI
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name qnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 1000 --seed 2512399976
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_qnli_accuracy(9_9_9).csv">IID(1:1:1) result</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_qnli_accuracy(19_6_1).csv">IID(19:6:1) result</a></li>
  </ul>
</details>



<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name qnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.027 --leader_epoch 5 --num_users 1000 --seed 2512399976 --manual_distribution
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_qnli_accuracy_noniid.csv">Non-IID result</a></li>
  </ul>
</details>


## MNLI
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name mnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.0027 --leader_epoch 5 --num_users 10000 --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --seed 3301259171
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_mnli_accuracy(9_9_9).csv">IID(1:1:1) result</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_mnli_accuracy(19_6_1).csv">IID(19:6:1) result</a></li>
  </ul>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name mnli --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.0027 --leader_epoch 5 --num_users 10000 --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --seed 3301259171 --manual_distribution
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_mnli_accuracy_noniid.csv">Non-IID result</a></li>
  </ul>
</details>



## MRPC
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name mrpc --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --seed 1046058099
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_mrpc_accuracy(9_9_9).csv">IID(1:1:1) result</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_mrpc_accuracy(19_6_1).csv">IID(19:6:1) result</a></li>
  </ul>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name mrpc --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution --seed 1046058099
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_mrpc_accuracy_noniid.csv">Non-IID result</a></li>
  </ul>
</details>



## RTE
<details>
  <summary>IID</summary>
  <pre><code>python LIFE.py --model_name_or_path roberta-base --task_name rte --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 1 1 1 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --seed 1661086535
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_rte_accuracy(9_9_9).csv">IID(1:1:1) result</a></li>
    <p> For IID(19:6:1) please modify '--group 19 6 1' </p>
    <li><a href="./output_diff_datasets/LIFE_rte_accuracy(19_6_1).csv">IID(19:6:1) result</a></li>
  </ul>
</details>

<details>
  <summary>Non-IID</summary>
  <pre><code>python LIFE_noniid.py --model_name_or_path roberta-base --task_name rte --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 5 3 2 --pick_percentage 0.1 --rounds 100 --strategy split --sample_ratio 0.27 --leader_epoch 5 --num_users 100 --manual_distribution  --seed 1661086535
  </code></pre>
  <p> The link to the results of the experiment: </p>
  <ul>
    <li><a href="./output_diff_datasets/LIFE_rte_accuracy_noniid.csv">Non-IID result</a></li>
  </ul>
</details>


## Note on the requirements.txt file
<details>
  The requirements.txt file is generated using Google Colab.
  The necessary libraries are as follows: argparse,logging,pandas,matplotlib,torch,datasets,tqdm,transformers,accelerate,huggingface_hub.
  If you are running experiments on Google Colab, you need to ensure that the 'accelerate' and 'datasets' libraries are installed. You can do this by running the following commands in a Colab cell '!pip install accelerate datasets'
  
</details>
