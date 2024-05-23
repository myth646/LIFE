# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import numpy as np
import torch.optim as optim
from collections import Counter
import csv
import torch
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from copy import deepcopy

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    set_seed,
)
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from server_cuda_FL2 import Aggregator, Aggregator_mom, process_model_grad, HeteAgg, HeteAgg_mom, HeteAgg_mom_FL2, evaluate, get_layer_id




logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        '--project',
        type=str,
        default='HeteFL',
        help='default name for HeteFL'
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs='+',
        default=[2e-4, 5e-5, 5e-5],
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.", default=None)
    #three heterogeneity   
    parser.add_argument("--num_hidden_layers", type=int, nargs='+', default=[4, 8, 12], help="The number of layers for each portion of users")
    #two heterogeneity
    # parser.add_argument("--num_hidden_layers", type=int, nargs='+', default=[4, 8], help="The number of layers for each portion of users")
    parser.add_argument("--local_cls", action="store_true")
    parser.add_argument("--local_one", action="store_true")
    parser.add_argument("--local_pooler", action="store_true")

    parser.add_argument("--portion", nargs='+', type=float, default=None, help='The portion for each group of users')
    parser.add_argument("--group_proportions", nargs='+', type=float, default=[0.2, 0.3, 0.5], help='Proportion of samples held by each cluster')
    parser.add_argument("--num_users", type=int, default=1000, help='number of users')
    parser.add_argument("--sample_ratio", type=float, default=0.027, help='users sample ratio per round')

    parser.add_argument("--rounds", type=int, default=30, help='rounds of FL')
    parser.add_argument("--save_per_round", type=bool, default=False, help='save per round')
    parser.add_argument("--log_round", type=int, default=None, help='log round')
    parser.add_argument("--cache_dir", type=str, default="../cache", help="cache data dir.")

    parser.add_argument("--mom_beta", type=float, default=0.2, help="momentum factor for larger model gradient")
    parser.add_argument("--mom_grad", action="store_true", help='turn on momentum gradient')
    parser.add_argument("--drop_idx", type=int, nargs='+', default=[], help='isolate device id for ablation study')
    parser.add_argument("--pick_percentage", type=float, default=0.1, help='Proportion of data contributed by leaders')
    parser.add_argument("--strategy", type=str, default='split', help='Leaders strategies for using LocalData : split, same, partial')
    parser.add_argument("--leader_epoch", type=int, default=5, help='Leaders in operation')
    parser.add_argument("--allweight", action="store_true", help='1 or all weight')
    parser.add_argument("--manual_distribution", action="store_true", help='Manual distribution processing')
    parser.add_argument("--dirichlet_alpha", nargs='+', type=float, default=[0.2, 0.5, 1], help='Distribution of dirichlet')

    parser.add_argument("--group", nargs='+', type=float, default=[9,9,9], help='The portion for each group of users')

    args = parser.parse_args()
    args.sample_ratio =[args.group[0]/(args.num_users*args.portion[0]/sum(args.portion)), args.group[1]/(args.num_users*args.portion[1]/sum(args.portion)), args.group[2]/(args.num_users*args.portion[2]/sum(args.portion))]
    print(args.sample_ratio)
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.log_round is None:
        args.log_round = math.ceil(1 / args.sample_ratio)

    # args.learning_rate = [float(x) for x in args.learning_rate]

    # read external config file
    # with open(f'../task_configs/{args.task_name}.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # args = vars(args)
    # args.update(config)

    return args

import csv

def save_to_csv(data, filename):

    if not os.path.exists('output_diff_datasets'):
        os.makedirs('output_diff_datasets')
    filename = os.path.join('output_diff_datasets', filename)

    group1 = []
    group2 = []
    group3 = []
    for index, item in enumerate(data):
        if index % 3 == 0:
            group1.append(item)
        elif index % 3 == 1:
            group2.append(item)
        else:
            group3.append(item)

    combined_data = list(zip(group1, group2, group3))

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(combined_data)

def append_to_csv(file_path, file_name, seed):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['File Name', 'Seed'])

        writer.writerow([file_name, seed])

class CSVDataPrinter:
    def __init__(self, file_paths,name_file_path, seed):
        self.file_paths = file_paths
        self.name_file_path = name_file_path
        self.seed = seed

    def extract_name(self, file_path):
        match = re.search(r'\((.*?)\)', file_path)
        return match.group(1) if match else "Unknown"

    def generate_image_name(self, file_name, seed):
        file_name_without_ext = file_name.replace('.csv', '')
        return f"{file_name_without_ext}_seed_{seed}.png"


    def print_data(self, col_indices, sum_columns=False):
        plt.figure()

        name_data = pd.read_csv(self.name_file_path)
        for file_path in self.file_paths:
            data = pd.read_csv(file_path)
            label_name = self.extract_name(file_path)
            
            if self.seed in name_data['Seed'].values:
                row = name_data[name_data['Seed'] == self.seed]
                file_name = row.iloc[0, 0]
                seed = row.iloc[0, 1]
            if sum_columns:
                sum_data = data.iloc[:, col_indices].mean(axis=1)
                plt.plot(sum_data, label=label_name)
            else:
                
                for col_index in col_indices:
                    if col_index < len(data.columns):
                        prefix = 'S' if col_index == 0 else 'M'
                        plt.plot(data.iloc[:, col_index], label=f"{prefix}{label_name}")
        plt.xlabel('ROUNDS')
        plt.ylabel('ACCURACY')
        plt.legend()
        # plt.show()

        folder_path = 'figure output'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        image_name = self.generate_image_name(file_name, seed)
        plt.savefig(os.path.join(folder_path, image_name))
        plt.close()


def leader_data_selection(i,uid2_sids,pick_count,leaders,train_dataset,strategy):
    leader_data = []
    index = []
    if pick_count == 0:
        train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]])
        leader_data.append(train_dataset_i)
        index.append(i)
    else:
        if strategy == 'split':
            random.shuffle(uid2_sids[leaders[i-1]])
            train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]][:pick_count])
            leader_data.append(train_dataset_i)
            index.append(i-1)
            train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]][pick_count:])
            leader_data.append(train_dataset_i)
            index.append(i)
        elif strategy == 'same':
            random.shuffle(uid2_sids[leaders[i-1]])
            train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]][:pick_count])
            leader_data.append(train_dataset_i)
            index.append(i-1)
            train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]][:pick_count])
            leader_data.append(train_dataset_i)
            index.append(i)
        elif strategy == 'partial':
            random.shuffle(uid2_sids[leaders[i-1]])
            train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]][:pick_count])
            leader_data.append(train_dataset_i)
            index.append(i-1)
            train_dataset_i = Subset(train_dataset, uid2_sids[leaders[i-1]])
            leader_data.append(train_dataset_i)
            index.append(i)
    return leader_data, index 

def generate_dirichlet_distributions_for_types(num_intervals,alpha_value):
    interval_ratios = {}
    for type_id, alpha in enumerate(alpha_value):
        alpha_vector = [alpha] * num_intervals
        proportions = np.random.dirichlet(alpha_vector)
        interval_ratios[type_id] = proportions.tolist()
    return interval_ratios

def distribute_non_iid_data_for_regression(dataset, uid2_dids, args):
    labels = dataset['labels']
    indices = list(range(len(labels)))
    threshold_25 = np.percentile(labels, 25)
    threshold_75 = np.percentile(labels, 75)
    low_idx = [idx for idx, label in zip(indices, labels) if label <= threshold_25]
    mid_idx = [idx for idx, label in zip(indices, labels) if threshold_25 < label <= threshold_75]
    high_idx = [idx for idx, label in zip(indices, labels) if label > threshold_75]

    # Distribution strategy
    if not args.manual_distribution:
        interval_ratios = generate_dirichlet_distributions_for_types(3, args.dirichlet_alpha)
    else:
        interval_ratios = {0: [7, 2, 1], 1: [2, 4, 4], 2: [1, 1, 1]}
    uid2_sids = [[] for _ in range(args.num_users)]

    # Calculate the number of samples that should be assigned to each user based on the user type
    for type_id in range(args.num_hete_total):
        type_user_indices = [j for j, x in enumerate(uid2_dids) if x == type_id]
        num_users_in_type = len(type_user_indices)
        total_samples = int(args.group_proportions[type_id] * len(indices))
        num_samples_per_user = total_samples // num_users_in_type

        # Calculate the number of samples each user should get in each interval
        num_low_per_user = int((interval_ratios[type_id][0] / sum(interval_ratios[type_id])) * num_samples_per_user)
        num_mid_per_user = int((interval_ratios[type_id][1] / sum(interval_ratios[type_id])) * num_samples_per_user)
        num_high_per_user = num_samples_per_user - num_low_per_user - num_mid_per_user

        for user_id in type_user_indices:
            if num_low_per_user <= len(low_idx):
                selected_low = random.sample(low_idx, num_low_per_user)
                low_idx = [idx for idx in low_idx if idx not in selected_low]
            else:
                selected_low = random.choices(low_idx, k=num_low_per_user)
                
            if num_mid_per_user <= len(mid_idx):
                selected_mid = random.sample(mid_idx, num_mid_per_user)
                mid_idx = [idx for idx in mid_idx if idx not in selected_mid]
            else:
                selected_mid = random.choices(mid_idx, k=num_mid_per_user)

            if num_high_per_user <= len(high_idx):
                selected_high = random.sample(high_idx, num_high_per_user)
                high_idx = [idx for idx in high_idx if idx not in selected_high]
            else:
                selected_high = random.choices(high_idx, k=num_high_per_user)

            uid2_sids[user_id].extend(selected_low)
            uid2_sids[user_id].extend(selected_mid)
            uid2_sids[user_id].extend(selected_high)

    return uid2_sids


def manual_distribute_data(dataset, uid2_dids, num_hete_total, group_proportions):
    """
    Distribute datasets to clients non-independently using Dirichlet distribution.

    :param dataset: Dataset containing features and labels
    :param uid2_dids: Client's array of cluster IDs
    :param num_hete_total: Total number of groups
    :param group_proportions: Proportions of the dataset for each group
    :param dirichlet_alpha: Alpha parameter for the Dirichlet distribution
    :return: Data index assigned to each client
    """
    unique_labels = set(data_point['labels'] for data_point in dataset)
    num_classes = len(unique_labels)

    data_by_label = {label_id: [i for i, data_point in enumerate(dataset) if data_point['labels'] == label_id] for label_id in unique_labels}
    data_by_label_backup = deepcopy(data_by_label)

    uid2_sids = [None] * len(uid2_dids)
    distribution_list = []
    for group_id in range(num_hete_total) :
        group_clients = [i for i, did in enumerate(uid2_dids) if did == group_id]
        total_samples_per_group = int(len(dataset) * group_proportions[group_id])
        if num_classes == 3:
            # Allocate different proportions of labels based on clusters
            if group_id == 0:  # Small clusters, predominantly labelled 0, less 1 and 2
                class_distribution = [0.7, 0.2, 0.1]
            elif group_id == 1:  # Medium cluster, labels 0 and 1 predominate, less 2
                class_distribution = [0.2, 0.4, 0.4]
                # class_distribution = [0.2, 0.3, 0.5]
            else:  # Large clusters, relatively balanced 0, 1, 2 tags
                class_distribution = [1/3, 1/3, 1/3]
        elif num_classes == 2:
            if group_id == 0:  # Small clusters, predominantly labelled 0, less 1 and 2
                class_distribution = [0.8, 0.2]
            elif group_id == 1:  # Medium cluster, labels 0 and 1 predominate, less 2
                class_distribution = [0.4, 0.6]
            else:  # Large clusters, relatively balanced 0, 1, 2 tags
                class_distribution = [0.5, 0.5]
        distribution_list.append(class_distribution)

        for label_id, class_prop in enumerate(class_distribution):
            num_samples_per_label = max(int(class_prop * total_samples_per_group), len(group_clients))

            available_samples = len(data_by_label[label_id])
            if available_samples < num_samples_per_label:
                group_samples = np.random.choice(data_by_label[label_id], available_samples, replace=False).tolist()              
                data_by_label[label_id] = [x for x in data_by_label[label_id] if x not in group_samples]
                additional_samples = num_samples_per_label - available_samples
                group_samples.extend(np.random.choice(data_by_label_backup[label_id], additional_samples, replace=True).tolist())
            else:
                group_samples = np.random.choice(data_by_label[label_id], num_samples_per_label, replace=False).tolist()              
                data_by_label[label_id] = [x for x in data_by_label[label_id] if x not in group_samples]

            samples_per_client = num_samples_per_label // len(group_clients)
            
            for client in group_clients:
                if uid2_sids[client] is None:
                    uid2_sids[client] = group_samples[:samples_per_client]
                else:
                    uid2_sids[client].extend(group_samples[:samples_per_client])  
                group_samples = group_samples[samples_per_client:]
    print(distribution_list)
    return uid2_sids

def dirichlet_distribute_data(dataset, uid2_dids, num_hete_total, group_proportions, dirichlet_alpha):
    """
    Distribute datasets to clients non-independently using Dirichlet distribution.

    :param dataset: Dataset containing features and labels
    :param uid2_dids: Client's array of cluster IDs
    :param num_hete_total: Total number of groups
    :param group_proportions: Proportions of the dataset for each group
    :param dirichlet_alpha: Alpha parameter for the Dirichlet distribution
    :return: Data index assigned to each client
    """
    unique_labels = set(data_point['labels'] for data_point in dataset)
    num_classes = len(unique_labels)

    data_by_label = {label_id: [i for i, data_point in enumerate(dataset) if data_point['labels'] == label_id] for label_id in unique_labels}
    data_by_label_backup = deepcopy(data_by_label)

    uid2_sids = [None] * len(uid2_dids)
    distribution_list = []
    for group_id in range(num_hete_total) :
        group_clients = [i for i, did in enumerate(uid2_dids) if did == group_id]
        total_samples_per_group = int(len(dataset) * group_proportions[group_id])

        class_distribution = np.random.dirichlet([dirichlet_alpha[group_id]] * num_classes)
        distribution_list.append(class_distribution)

        for label_id, class_prop in enumerate(class_distribution):
            num_samples_per_label = max(int(class_prop * total_samples_per_group), len(group_clients))

            available_samples = len(data_by_label[label_id])
            if available_samples < num_samples_per_label:
                group_samples = np.random.choice(data_by_label[label_id], available_samples, replace=False).tolist()              
                data_by_label[label_id] = [x for x in data_by_label[label_id] if x not in group_samples]
                additional_samples = num_samples_per_label - available_samples
                group_samples.extend(np.random.choice(data_by_label_backup[label_id], additional_samples, replace=True).tolist())
            else:
                group_samples = np.random.choice(data_by_label[label_id], num_samples_per_label, replace=False).tolist()              
                data_by_label[label_id] = [x for x in data_by_label[label_id] if x not in group_samples]

            samples_per_client = num_samples_per_label // len(group_clients)
            
            for client in group_clients:
                if uid2_sids[client] is None:
                    uid2_sids[client] = group_samples[:samples_per_client]
                else:
                    uid2_sids[client].extend(group_samples[:samples_per_client])  
                group_samples = group_samples[samples_per_client:]
    print(distribution_list)
    return uid2_sids


def main():
    args = parse_args()

    # set up FL users
    args.num_hete_total = len(args.num_hidden_layers)
    if args.portion is None:
        args.portion = (np.ones(args.num_hete_total) * (1.0 / args.num_hete_total)).tolist() # XXX: draw from a uniform distribution => draw from 3 gaussian distributions with left/central/right mean

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerators = []
    for i in range(args.num_hete_total):
        accelerators.append(Accelerator())
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerators[0].state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerators[0].is_local_main_process else logging.ERROR)
    if accelerators[0].is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    else:
        args.seed = int.from_bytes(os.urandom(4), 'big')
        set_seed(args.seed)

    # Handle the repository creation
    if accelerators[0].is_main_process:
        if args.push_to_hub:
            # if args.hub_model_id is None and args.hub_token is not None:
            #     repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            # else:
            repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerators[0].wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # dummy model
    dummy_models = []
    dummy_names = []
    leave_one_names = []
    for i in range(args.num_hete_total):
        # config
        config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                            num_labels=num_labels, 
                                            finetuning_task=args.task_name,
                                            num_hidden_layers= args.num_hidden_layers[i])

        # model
        model = AutoModelForSequenceClassification.from_pretrained(
                            args.model_name_or_path,
                            from_tf=bool(".ckpt" in args.model_name_or_path),
                            config=config,
                            )
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logging.info("Num of parameters for model {} = {}".format(i, model_size))

        dummy_models.append(model)
        dummy_names.append([name for name, param in model.named_parameters() if param.requires_grad])

        leave_one_names.append([name for name in dummy_names[i] if get_layer_id(name) == args.num_hidden_layers[i] - 1])

        del model
        del config

    logger.info(dummy_names)

    # hete exclude
    hete_exclude_sub = []
    if args.local_cls:
        hete_exclude_sub.append('classi')
        logger.info('local (no share) classifier')
    if args.local_pooler:
        hete_exclude_sub.append('pooler')
        logger.info('local (no share) pooler')
    hete_exclude_name = []
    for layer_name in dummy_names[-1]:
        for sub in hete_exclude_sub:
            if sub in layer_name:
                hete_exclude_name.append(layer_name) 
                logger.info(f"Exclude parameter {layer_name}")

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        dummy_models[0].config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in dummy_models[0].config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        dummy_models[0].config.label2id = label_to_id
        dummy_models[0].config.id2label = {id: label for label, id in dummy_models[0].config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        dummy_models[0].config.label2id = {l: i for i, l in enumerate(label_list)}
        dummy_models[0].config.id2label = {id: label for label, id in dummy_models[0].config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerators[0].main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerators[0].use_fp16 else None))

    # HACK: global set up
    num_train_sam = [int(len(train_dataset) * prop) for prop in args.group_proportions]
    aggs = []
    
    for i in range(args.num_hete_total):
        if not args.mom_grad:
            aggs.append(Aggregator(i, args, num_labels, num_train_sam))
        else:
            aggs.append(Aggregator_mom(i, args, num_labels, num_train_sam))

    total_batch_sizes = [args.per_device_train_batch_size * accelerators[i].num_processes for i in range(args.num_hete_total)]
    uid2_dids = random.choices(population=range(args.num_hete_total),
                            weights=args.portion,
                            k=args.num_users)
    tmp_cnt = Counter(uid2_dids)

    zero_indices = [i for i, x in enumerate(uid2_dids) if x == 0]
    one_indices = [i for i, x in enumerate(uid2_dids) if x == 1]
    two_indices = [i for i, x in enumerate(uid2_dids) if x == 2]

    set_seed(args.seed+1)
    if not is_regression:
        if not args.manual_distribution:
            uid2_sids = dirichlet_distribute_data(train_dataset, uid2_dids, args.num_hete_total, args.group_proportions, args.dirichlet_alpha)
        else:
            uid2_sids = manual_distribute_data(train_dataset, uid2_dids, args.num_hete_total, args.group_proportions)
    else:
        uid2_sids = distribute_non_iid_data_for_regression(train_dataset,uid2_dids, args)
    
    logger.info("***** Running training *****")
    logger.info(f"  seed = {args.seed}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_sizes}")
    logger.info(f"  Portion of hete devices acorss {args.num_users} = {tmp_cnt}")
    logger.info(f"  Number of samples per group= {num_train_sam}")
    product = [int(n * r) for n, r in zip(num_train_sam, args.sample_ratio)]
    logger.info(f"  Sample ratio per round & groups = {args.sample_ratio}, with {product} samples.")

    if args.task_name in ['mrpc', 'qqp']:
        eval_name = ["accuracy","f1"]
    elif args.task_name == 'cola':
        eval_name = ["matthews_correlation"]
    elif args.task_name == 'stsb':
        eval_name = ['spearmanr', 'pearson']
    else:
        eval_name = ["accuracy"]

    best_eval = {}
    best_eval_r = {}
    for e_name in eval_name:
        best_eval.update(
            {
                f'eval-u{i}/best_{e_name}': 0 for i in range(args.num_hete_total)
            }
        )
        best_eval_r.update(
            {
                f'eval-u{i}/best_{e_name}_r': 0 for i in range(args.num_hete_total) 
            }
        )

    # eval and metric
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size) # XXX(1): split evel dataset (if non-iid) for personalized layer test
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")


    selected_zero_index = random.sample(zero_indices, int(args.group[0]))
    selected_one_index = random.sample(one_indices, int(args.group[1]))
    selected_two_index = random.sample(two_indices, int(args.group[2]))
    selected_Mleader_index = random.sample(selected_one_index, 1)
    selected_Lleader_index = random.sample(selected_two_index, 1)
    leaders=selected_Mleader_index+selected_Lleader_index

    users = selected_zero_index + selected_one_index + selected_two_index
    random.shuffle(users)
    
    did2_sids = [[] for _ in range(args.num_hete_total)]
    for u in users:
        u_did = uid2_dids[u]
        did2_sids[u_did].extend(uid2_sids[u])

        # get sample portion for each type of device for current round
    did2_sample_portion = []
    skip_did = []
    for did in range(args.num_hete_total):
        if len(did2_sids[did]) < 1:
            skip_did.append(did)
        did2_sample_portion.append(len(did2_sids[did]))
        # logging.info(f"********** sample portion for r={r} is {did2_sample_portion}")
    tot = sum(did2_sample_portion)
    did2_sample_portion = [x/tot for x in did2_sample_portion]
    if args.allweight:
        did2_sample_portion1 = did2_sample_portion.copy()
        did2_sample_portion1[2] = 0  
        tot1 = sum(did2_sample_portion1)  
        proportions1 = [x / tot1 if i != 2 and tot1 else 0 for i, x in enumerate(did2_sample_portion1)]  
        did2_sample_portion2 = did2_sample_portion.copy()
        did2_sample_portion2[0] = 0  
        tot2 = sum(did2_sample_portion2)  
        proportions2 = [x / tot2 if i != 0 and tot2 else 0 for i, x in enumerate(did2_sample_portion2)] 
    else: 
        did2_sample_portion1 = did2_sample_portion.copy()
        did2_sample_portion1[2] = 0 
        did2_sample_portion1[1] = len(uid2_sids[leaders[0]])
        tot1 = sum(did2_sample_portion1)  
        proportions1 = [x / tot1 if i != 2 and tot1 else 0 for i, x in enumerate(did2_sample_portion1)]  
        did2_sample_portion2 = did2_sample_portion.copy()
        did2_sample_portion2[0] = 0
        did2_sample_portion2[2] = len(uid2_sids[leaders[1]]) 
        tot2 = sum(did2_sample_portion2)  
        proportions2 = [x / tot2 if i != 0 and tot2 else 0 for i, x in enumerate(did2_sample_portion2)]

    proportions = [proportions1, proportions2]
    print(proportions)

    eval_metrics_list = [[] for _ in range(len(eval_name))]
    for r in tqdm(range(args.rounds)):
        for idx in range(args.num_hete_total):
            train_dataset_i = Subset(train_dataset, did2_sids[idx])
            train_dataloader = DataLoader(
                                train_dataset_i, 
                                shuffle=True, 
                                collate_fn=data_collator,  
                                batch_size=args.per_device_train_batch_size
            )
            dummy_models[idx], aggs[idx].model, aggs[idx].optimizer, train_dataloader, eval_dataloader = accelerators[idx].prepare(
                dummy_models[idx], aggs[idx].model, aggs[idx].optimizer, train_dataloader, eval_dataloader
            )
            dummy_models[idx].train()
            dummy_models[idx].load_state_dict(aggs[idx].model.state_dict())
            for step, batch in enumerate(train_dataloader):
                # dummy_models[i].load_state_dict(aggs[i].model.state_dict())
                dummy_optimizer = optim.SGD(dummy_models[idx].parameters(), lr=args.learning_rate[idx])
                outputs = dummy_models[idx](**batch)
                loss = outputs.loss
                dummy_optimizer.zero_grad()
                loss.backward()
                model_grad = process_model_grad(dummy_models[idx].named_parameters(), args.per_device_train_batch_size / len(did2_sids[idx]))
                aggs[idx].collect(model_grad)
            aggs[idx].update()
            if args.task_name in ['mrpc', 'qqp','stsb']:
                result1, result2 = evaluate(idx, r, args, aggs[idx].model, accelerators[idx], eval_dataloader, metric, is_regression, logger, best_eval, best_eval_r, loss, processed_datasets, data_collator)
                eval_metrics_list[0].append(result1)
                eval_metrics_list[1].append(result2)
            else:
                result = evaluate(idx, r, args, aggs[idx].model, accelerators[idx], eval_dataloader, metric, is_regression, logger, best_eval, best_eval_r, loss, processed_datasets, data_collator)
                eval_metrics_list[0].append(result)

        if (r+1) % args.leader_epoch == 0:
            for i in range(len(leaders),0,-1):
                if i == 2:
                    args.drop_idx.append(0)
                    top = True
                else:
                    args.drop_idx.append(2)
                    top = False
                    
                if args.pick_percentage == 0:
                    pick_count = 0
                else:
                    pick_count = max(1,int(len(uid2_sids[leaders[i-1]]) * args.pick_percentage))
                    unpick_count = len(uid2_sids[leaders[i-1]]) - pick_count
                    
                first_iteration =True
                for leader_data, idx in  zip(*leader_data_selection(i,uid2_sids,pick_count,leaders,train_dataset,args.strategy)):
                    train_dataloader = DataLoader(
                                        leader_data, 
                                        shuffle=True, 
                                        collate_fn=data_collator,  
                                        batch_size=args.per_device_train_batch_size
                    )

                    dummy_models[idx], aggs[idx].model, aggs[idx].optimizer, train_dataloader, eval_dataloader = accelerators[idx].prepare(
                        dummy_models[idx], aggs[idx].model, aggs[idx].optimizer, train_dataloader, eval_dataloader
                    )

                    dummy_models[idx].train()
                    dummy_models[idx].load_state_dict(aggs[idx].model.state_dict())

                    for step, batch in enumerate(train_dataloader):
                        # dummy_models[i].load_state_dict(aggs[i].model.state_dict())
                        dummy_optimizer = optim.SGD(dummy_models[idx].parameters(), lr=args.learning_rate[idx])
                        outputs = dummy_models[idx](**batch)
                        loss = outputs.loss
                        dummy_optimizer.zero_grad()
                        loss.backward()
                        if first_iteration:
                            model_grad = process_model_grad(dummy_models[idx].named_parameters(), 1.0 / pick_count)
                        else:
                            model_grad = process_model_grad(dummy_models[idx].named_parameters(), 1.0 / unpick_count)
                        aggs[idx].collect(model_grad)
                    aggs[idx].update()
                if not args.mom_grad:
                    HeteAgg(r, logger, args, aggs, dummy_names, leave_one_names, hete_exclude_name, did2_sample_portion)
                else:
                    if top:
                        HeteAgg_mom_FL2(r, logger, args, aggs, dummy_names, leave_one_names, hete_exclude_name, proportions[1],i-1)
                    else:
                        HeteAgg_mom_FL2(r, logger, args, aggs, dummy_names, leave_one_names, hete_exclude_name, proportions[0],i-1)
                args.drop_idx =[]
        
            

    number_str = ''.join(str(num) for num in args.group)    
    for i in range(len(eval_metrics_list)):
        save_to_csv(eval_metrics_list[i], f'LIFE_{args.task_name}_{eval_name[i]}_noniid.csv')
        append_to_csv('output_seed.csv', f'LIFE_{args.task_name}_{eval_name[i]}_noniid.csv', args.seed)



if __name__ == "__main__":
    main()
# python LIFE.py --model_name_or_path roberta-base --task_name sst2 --local_cls --local_one --local_pooler --mom_grad --mom_beta 0.2 --log_round 5 --portion 3 2 1 --pick_percentage 0.1 --rounds 100 --strategy same --leader_epoch 5
# python test_cuda_out_all.py --model_name_or_path roberta-base --task_name sst2 --local_cls --local_one --local_pooler --mom_beta 0.2 --log_round 5 --portion 3 2 1 --pick_percentage 0.1 --strategy same --leader_epoch 5
