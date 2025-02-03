## Overview
This repository contains the code to reproduce the finetuning of Mistral 7B and LLaMA 3.1 8B on the earnings call dataset.

## Dependencies
- python 3.11
- pytorch 2.5.0 (cu124)
- transformers 4.46.3
- datasets 3.1.0
- unsloth 2024.12.2
- polars 1.16.0

> Note 1: To speed up training and reduce memory usage, we use Unsloth for the finetuning and inference. The installation of Unsloth is a bit tricky, depending on the CUDA version, pytorch version, and GPU model. Please refer to [Unsloth's documentation](https://github.com/unslothai/unsloth) for the detailed instructions.

> Note 2: The code is only tested on the above dependencies. It may not work on other versions.

> Note 3: Compared with a python script, training on a notebook is about 20% slower. But for readability and interactivity, we use a notebook in this repository.

## How to Run
### Prepare the Data
- Download the training and test data using [this link](https://1drv.ms/f/s!AqUu9ylMgqcDg4LCHRr6-KiTv4GoCOc?e=xsoL1f). Put the downloaded files (`test_22q*.feather` and `train_22q*.feather`) in a new folder `data` and put the `data` folder under project directory.

### Finetune the Model
Run `finetune.py` to finetune the model. For example, to finetune Mistral 7B on time period 20Q4-22Q3 and test on 22Q4, run
```python
python finetune.py \
    --model_name unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
    --train_data_path data/train_22q4.feather \
    --test_data_path data/test_22q4.feather
```

Finetuning on one window takes about 9 hours on RTX 4900.

The finetuned model will be saved in the `saved_results/` folder.
- For example, the finetuned Mistral 7B model whose test period is 22Q4 will be saved in `saved_results/mistral_22q4`.

### Inference Using the Finetuned Model
Run `inference_unsloth.ipynb` to inference using the finetuned model. The arguments are detailed in the notebook. The most important arguments are:
- `model_name`: The name of the finetuned model.
- `test_data_path`: The path to the test data.

The inference results will be saved in the `saved_results/` folder.
- e.g., `saved_results/results_llama-3.1_noft_frtxt_22q3.feather` for Llama 3.1 8B, non-finetuned, using financial ratios and text features, and test on 22Q3.
- e.g., `saved_results/results_mistral_ft_frtxt_22q4.feather` for Mistral 7B, finetuned, using financial ratios and text features, and test on 22Q4.

### Benchmark
Run `benchmark.ipynb` to benchmark the finetuned models. The arguments are detailed in the notebook.


