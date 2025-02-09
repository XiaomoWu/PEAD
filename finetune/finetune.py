# %%
# import required libraries
import logging
import os
import re
import textwrap
import argparse

import polars as pl
import torch
from datasets import Dataset
from polars import col as c
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# Configure logging to show timestamp, log level and message
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S %p",
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    help="Name of the model to use. Use 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit' for Mistral 7B "
    "or 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit' for Llama 3.1 8B",
)
parser.add_argument(
    "--train_data_path", type=str, default="data/train_22q4.feather", help="Path to training data"
)
parser.add_argument(
    "--test_data_path", type=str, default="data/test_22q4.feather", help="Path to test data"
)
args = parser.parse_args()

# %% [markdown]
# ## Configuration

model_name = args.model_name
max_seq_length = 20000
max_new_tokens = 8

# device
device = f"cuda:0"

# use finratio or not
use_finratio = True

# max training steps
# set to -1 for full training
max_steps = -1

# data paths
train_data_path = args.train_data_path
test_data_path = args.test_data_path

# sanity check if `chat_template` matches `model_name`
# by checking if `llama` or "mistral" appear in both
if "llama" in model_name.lower():
    chat_template = "llama-3.1"
elif "mistral" in model_name.lower():
    chat_template = "mistral"
else:
    raise ValueError("model_name must contain 'llama' or 'mistral'")

# saved model name
split_id = re.search(r"(\d{2}q\d)", test_data_path).group(1)
saved_model_name = f"saved_results/{chat_template}_{split_id}"


# %% [markdown]
# ## Prepare Data

# %% [markdown]
# Import data for training

# %%
# read the huggingface dataset
train_data = pl.read_ipc(train_data_path)
test_data = pl.read_ipc(test_data_path)
logging.info(f"Read data from {train_data_path} and {test_data_path}")


# %% [markdown]
# Next, we construct prompts using the training and testing data. The output consists of two huggingface datasets: `train_dataset` and `test_dataset`.

# %%
# get tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = get_chat_template(
    tokenizer,
    chat_template=chat_template,
    map_eos_token=True,  # e.g., maps <|im_end|> to </s> instead
)


# construct the prompt for training data
def construct_prompt(dataset, use_finratio):
    output = []
    for row in dataset.iter_rows(named=True):
        # define system
        prompt_transcript = textwrap.dedent(f"""
            You are a financial analyst. You will be given an earnings call transcript of a company. It may contain three parts: Management Discussion, Questions from Analysts, and Answers from Management. The "Management Discussion" section is a statement from the management (usually CEOs and CFOs) about the past performance and future prospects of the company. The "Questions from Analysts" part consists of questions from financial analysts, and the "Answers from Management" part contains responses from the management. There may be multiple rounds of questions and answers in a call. Your task is to predict whether the earnings call will have a positive or negative impact on the future stock return. Please answer by typing an integer score in the range of 1 to 5, where 1 is the most negative and 5 is the most positive. Your reply must start with "Score:N". Please do not concentrate your predictions on the same score, i.e., the number of stocks falling into each score should be balanced. Please also note that the management usually uses very positive language when discussing their company, but you should not take it for granted. Pay attention to the questions from the analysts. 
                    
            Now the earnings call transcript begins:
                    
            {row["text"]}
        """).strip()

        prompt_finratio = textwrap.dedent(f"""
            Now the financial ratios begin:
            - Earnings surprise (normalized by stock price): {row["sue3"]}
            - Return volatility in the past month: {row["vol_call_m21_m1"]}
            - Market capitalization (log-transformed): {row["mcap"]}
            - Book-to-market ratio: {row["bm"]}
            - Return-on-assets: {row["roa"]}
            - Debt-to-assets: {row["debt_assets"]}
            - Median earnings forecast: {row["medest"]}
            - Number of analysts forecasting: {row["numest"]}
            - Standard deviation of earnings forecast: {row["stdest"]}
            - Turnover in the past month: {row["turnover_ma21"]}
            - Trading volume in the past month: {row["volume_ma21"]}
        """).strip()

        if use_finratio:
            user_prompt = prompt_transcript + "\n\n" + prompt_finratio
        else:
            user_prompt = prompt_transcript

        # define user and assistant messages
        # define user and assistant messages
        user = {
            "transcriptid": row["transcriptid"],
            "docid": row["docid"],
            "docid_idx": row["docid_idx"],
            "role": "user",
            "content": user_prompt,
        }
        assistant = {
            "transcriptid": row["transcriptid"],
            "docid": row["docid"],
            "docid_idx": row["docid_idx"],
            "role": "assistant",
            "content": f'Score:{row["rank"]}',
        }

        output.append([user, assistant])

    # convert to HF Dataset
    dataset = Dataset.from_dict({"chat": output})

    # apply chat template
    dataset = dataset.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template(
                x["chat"], tokenize=False, add_generation_prompt=False
            )
        },
        batched=True,
    )

    return dataset


# construct the prompt for training and testing data
train_dataset = construct_prompt(train_data, use_finratio)
test_dataset = construct_prompt(test_data, use_finratio)

# %% [markdown]
# ## Prepare Model and Tokenizer

# %% [markdown]
# Initial model and tokenizer

# %%
# Get model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,  # None for autodetect
    load_in_4bit=True,
    device_map=device,
)

# add LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=985,
    use_rslora=True,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


# %% [markdown]
# ## Train

# %%
# init trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    dataset_num_proc=16,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        # warmup_steps=10,
        warmup_ratio=0.1,
        max_steps=max_steps,  # set to -1 for full training
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        report_to="none",
        eval_steps=100,
        optim="adamw_8bit",  # adamw_8bit
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=985,
        output_dir=saved_model_name,
        save_strategy="no",
        overwrite_output_dir=True,
    ),
)

# %%
# (important!) mask the user message so that the model is
# only trained on assistant responses!
# see this Unsloth instruction notebook for more details: https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing

if chat_template == "mistral":
    instruction_part = "[INST] "
    response_part = "[/INST] "
elif chat_template == "llama-3.1":
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

trainer = train_on_responses_only(
    trainer,
    instruction_part=instruction_part,
    response_part=response_part,
)

# %%
# print out the prompt before and after masking
# you should see that the user message is masked (i.e., empty string)
tokenizer.decode(trainer.train_dataset[0]["input_ids"])
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
tokenizer.decode(
    [space if x == -100 else x for x in trainer.train_dataset[0]["labels"]]
)

# %%
# train the model
logging.info(f"Training on {saved_model_name} | Split: {split_id} | Device: {device}")
trainer_stats = trainer.train()

# save the model
model.save_pretrained_merged(
    saved_model_name,
    tokenizer,
    save_method="lora",
)

logging.info(f"Saved to {saved_model_name} | Split: {split_id} | Device: {device}")
