import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# init the logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)


def set_wdir():
    """change working directory based on the machine"""

    if "yu-ws" in os.uname().nodename:
        wdir = Path("/home/yu/chaoyang/projects/Call/call")
    elif "lerner" in os.uname().nodename:
        wdir = Path("/home/yuzhu/synology/projects/Call/call")
    os.chdir(wdir)


set_wdir()


@dataclass
class Config:
    model_name: str
    max_seq_length: int
    finetune_or_inference: str
    train_data_path: str
    test_data_path: str
    saved_model_name: str
    chat_template: str
    max_new_tokens: int
    generated_text_path: str
    use_streamer: bool
    unsloth: bool
    run_id: str

    def __post_init__(self):
        # check if finetune_or_inference is valid
        if self.finetune_or_inference not in ["finetune", "inference"]:
            raise ValueError(
                "finetune_or_inference must be either 'finetune' or 'inference'"
            )

        # check if chat_template is valid
        if self.chat_template not in ["qwen-2.5", "llama-3.1"]:
            raise ValueError("chat_template must be one of 'qwen-2.5' or 'llama-3.1'")


def build_prompt(
    train_data_path,
    test_data_path,
    model_name,
    chat_template,
    finetune_or_inference,
    max_seq_length,
    unsloth,
    **kwargs,
):
    """Read the feather file and construct the prompt

    This feather file contains both call transcripts and the corresponding labels

    Args:
        feather_path (str): path to the feather file
        model_name (str): name of the model
        chat_template (str): chat template to use, e.g., chatml
        add_generation_prompt (bool): only True if you're doing inference
        unsloth (bool): only True if you're using unsloth

    Returns:
        dataset (Dataset): HF Dataset with the prompt in the "chat" column
    """

    def _build_prompt(data_path):
        # read the huggingface dataset
        data = pl.read_parquet(data_path)
        logging.info(f"Read data from {data_path}")

        # get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if unsloth:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=chat_template,
                map_eos_token=True,  # e.g., maps <|im_end|> to </s> instead
            )

        # construct the prompt
        # for finetuning, the output is a HF Dataset; for inference, it's a list of dicts
        dataset = []
        for row in data.iter_rows(named=True):
            # define system
            user_content = f"""
    You are a financial analyst. You will be given an earnings call transcript of a company and a few financial ratios. Your task is to predict whether the earnings call will have positive or negative impact on the future stock return. Please answer by typing a score between 1 and 5, where 1 is the least positive and 5 is the most positive. Your answer must start with "Score:". Please do not concentrate your predictions on the same score, i.e., the number of stocks falling to each score should be balanced. 
    The earnings call may contain three parts: Management Discussion, Questions from Analysts, and Answers from Management. The "Management Discussion" section is a statement from the management (usually CEOs and CFOs) about the past performance and future prospects of the company. The "Questions from Analysts part" is a question from financial analysts and the "Answers from Management part" is the response from the management. There may be multiple rounds of questions and answers in a call. Please also note that the management usually uses very positive language when discussing their company, but you should not take it as granted. Pay attention to the questions from the analysts. 

    Now the earnings call transcript begins:
    {row["text"]}

    Now the financial ratios begin:
    - Earnings surprise (normalized by stock price): {row["sue3"]}
    - Return volatility in the past month: {row["vol_call_m21_m1"]}
    - Market capitalization (log-transformed): {row["mcap"]}
    - Book-to-market ratio: {row["bm"]}
    - Return-on-assets: {row["roa"]}
    - Debt-to-assets: {row["debt_assets"]}
    - Median earnings forecast: {row["medest"]}
    - Number analysts forecast: {row["numest"]}
    - Standard deviation of earnings forecast: {row["stdest"]}
    - Turnover in the past month: {row["turnover_ma21"]}
    - Trading volume in the past month: {row["volume_ma21"]}
    """

            # define user and assistant
            user = {
                "docid": row["docid"],
                "role": "user",
                "content": user_content,
            }
            assistant = {
                "docid": row["docid"],
                "role": "assistant",
                "content": f'rank:{row["rank"]}',
            }

            # create prompt based on finetune_or_inference
            if finetune_or_inference == "finetune":
                dataset.append([user, assistant])
            elif finetune_or_inference == "inference":
                dataset.append([user])

        # convert to HF Dataset
        dataset = Dataset.from_dict({"chat": dataset})

        # if inference, return a list of numpy arrays (tokenized prompts)
        if finetune_or_inference == "inference":
            dataset = dataset.map(
                lambda x: {
                    "input_ids": tokenizer.apply_chat_template(
                        x["chat"],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="np",
                        max_length=max_seq_length,
                    )
                },
                batched=True,
                batch_size=8,
                drop_last_batch=False,
            )
            dataset.set_format(type="torch", columns=["input_ids"])

        # if finetuine, return a HF Dataset
        if finetune_or_inference == "finetune":
            # apply chat template
            dataset = dataset.map(
                lambda x: {
                    "prompt": tokenizer.apply_chat_template(
                        x["chat"], tokenize=False, add_generation_prompt=False
                    )
                },
                batched=True,
            )

        logging.info("Constructed the prompt")

        return dataset

    # build the prompt
    test_dataset = _build_prompt(test_data_path)
    train_dataset = None
    if finetune_or_inference == "finetune":
        train_dataset = _build_prompt(train_data_path)

    return train_dataset, test_dataset


def init_model_and_tokenizer(
    max_seq_length, model_name, finetune_or_inference, unsloth, **kwargs
):

    # get model and tokenizer
    if unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,  # None for autodetect
            load_in_4bit=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if finetune, add adapter
    if finetune_or_inference == "finetune":
        # add adapter
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                # "k_proj",
                "v_proj",
                # "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
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

    # if inference, prepare for inference
    if finetune_or_inference == "inference":
        if unsloth:
            FastLanguageModel.for_inference(model)
        else:
            model.eval()

    return model, tokenizer


def train(
    model_name,
    model,
    tokenizer,
    train_dataset,
    max_seq_length,
    saved_model_name,
    chat_template,
    val_dataset=None,
    **kwargs,
):
    # init trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="prompt",
        max_seq_length=max_seq_length,
        dataset_num_proc=16,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            # warmup_steps=10,
            warmup_ratio=0.1,
            max_steps=40,  # set to -1 for full training
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=1,
            eval_steps=5,
            optim="adamw_8bit",  # adamw_8bit
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=985,
            output_dir=f"saved_model",
            save_strategy="no",
            overwrite_output_dir=True,
        ),
    )

    # (important) only train on assistant responses!
    if "qwen" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        logging.info(f"Trained on responses only (template: {chat_template})")
    elif "llama" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        logging.info(f"Trained on responses only (template: {chat_template})")
    else:
        raise ValueError("model_name must contain 'qwen' or 'llama'")

    trainer_stats = trainer.train()

    # save the model
    model.save_pretrained_merged(
        saved_model_name,
        tokenizer,
        save_method="lora",
    )

    return trainer_stats

def inference(model, tokenizer, dataset, max_new_tokens, max_seq_length, use_streamer, generated_text_path, run_id, **kwargs):
    # debug only
    # dataset = dataset.select(range(2))

    # if using streamer
    if use_streamer:
        text_streamer = TextStreamer(tokenizer)
        for inputs in dataset["input_ids"]:
            _ = model.generate(
                input_ids=inputs.unsqueeze(0).to(model.device),
                streamer=text_streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        
        return

    # if not using streamer
    results = []
    for call, input_ids in tqdm(
        zip(dataset["chat"], dataset["input_ids"]), total=len(dataset)
    ):
        # Record the start time
        start_time = time.time()

        docid = call[0]["docid"]
        prompt = call[0]["content"]
        input_length = input_ids.shape[0]  # input_ids is a 1D tensor

        # debug only
        # if not (input_length >= 5000):
        #     logging.info(f"Skipping \"{docid}\" ({input_length} tokens)")
        #     continue

        generated_ids = model.generate(
            input_ids=input_ids.unsqueeze(0).to(model.device),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
        )
        generated_text = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )[0]

        # every output saved as a single file
        out = {"docid": docid, "prompt": prompt, "response": generated_text}
        out = pl.DataFrame(out)
        out.write_ipc(f"{generated_text_path}/{docid}.feather")

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"{docid} | {input_length} tokens | {elapsed_time:.2f} seconds")

    # at the end, read all the files and save as a feather file
    files = Path(generated_text_path).glob("*.feather")
    results = []
    for file in files:
        results.append(pl.read_ipc(file))

    results = pl.concat(results, how='vertical')
    results.write_ipc(f"{generated_text_path}.feather")

    # delete the temporary files
    shutil.rmtree(generated_text_path)

def main():
    # generate a unique id at the strat of each run
    run_id = str(uuid.uuid4())

    finetune_or_inference = "finetune"

    # hyperparameters
    config = Config(
        # for both finetuning and inference
        finetune_or_inference=finetune_or_inference,
        unsloth=True,
        run_id=run_id,
        model_name="unsloth/mistral-7b-bnb-4bit",  # "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "saved_model/conversation_qwen2.5_32b_batch01",
        max_seq_length=15000, 
        chat_template="llama-3.1",  # "llama-3.1" for llama, "qwen-2.5" for qwen2.5
        train_data_path="code/v4/reproduce-finetune/data/train_data.parquet",
        test_data_path="code/v4/reproduce-finetune/data/test_data.parquet",
        # for finetune only
        saved_model_name="code/v4/reproduce-finetune/saved_model/llama-3.1-8b",
        # for inference only
        generated_text_path="code/v4/reproduce-finetune/data/generated_text",
        use_streamer=False,
        max_new_tokens=1024,  # 16338
    )

    # build the prompt (as a HF Dataset)
    train_data, test_data = build_prompt(**asdict(config))

    # init model and tokenizer
    model, tokenizer = init_model_and_tokenizer(**asdict(config))

    # train
    if finetune_or_inference == "finetune":
        trainer_stats = train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            val_dataset=test_data,
            **asdict(config),
        )

    # inference
    if finetune_or_inference == "inference":
        inference(
            model=model,
            tokenizer=tokenizer,
            dataset=test_data,
            **asdict(config),
        )


if __name__ == "__main__":
    main()
