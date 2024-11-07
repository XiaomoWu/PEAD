## Overview
This repository contains the code to reproduce the finetuning of Mistral 7B on the earnings call dataset. The training data contain 6673 earnings calls from 2020-7-1 to 2022-6-30. The test data contains 1198 earnings calls from 2022-7-1 to 2023-9-30.

## Dependencies
- Python 3.11
- PyTorch 2.5.0
- Transformers 4.46.2
- Datasets 3.0.2
- Unsloth 2024.11.5

> Note 1: To speed up training and reduce memory usage, we use Unsloth for the finetuning and inference. The installation of Unsloth is a bit tricky, depending on the CUDA version, pytorch version, and GPU model. Please refer to [Unsloth's documentation](https://github.com/unslothai/unsloth) for the detailed instructions.

> Note 2: The code is tested on the above versions. It may not work on other versions.

## How to run
- Download the data using [this link](https://1drv.ms/f/s!AqUu9ylMgqcDg4LCHRr6-KiTv4GoCOc?e=OI0US7). Put the downloaded files (`train_data.parquet` and `test_data.parquet`) in a new folder `data` and put the `data` folder under project directory.

- Run `train.ipynb` and follow the instructions.
    - This notebook contains both the training and inference code.