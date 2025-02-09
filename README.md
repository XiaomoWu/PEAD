## Overview
This repository contains the code to reproduce 1) the GradPerp algorithm and 2) the finetuning of Mistral 7B and LLaMA 3.1 8B on the earnings call dataset.

## GradPerp

### Dependencies
Our project heavily relies on the [Hydra](https://hydra.cc/) library for configuration management. Please refer to the [Hydra documentation](https://hydra.cc/docs/intro/) for detailed instructions.

We use [Lightning](https://lightning.ai/) for training and evaluation. Please refer to the [Lightning documentation](https://lightning.ai/docs/pytorch/latest/) for detailed instructions.

We use [Deepspeed](https://www.deepspeed.ai/) for distributed training. Please refer to the [Deepspeed documentation](https://www.deepspeed.ai/docs/index.html) for detailed instructions.

We use [Wandb](https://wandb.ai/) for experiment logging and sweeping over multiple training/test windows. Please refer to the [Wandb documentation](https://docs.wandb.ai/index) for detailed instructions.

> [!IMPORTANT] 
> To successfully run the code, please register an account on [Wandb](https://wandb.ai/) and get the API key. The API key can be found at [this link](https://wandb.ai/authorize). After installing wandb, **please run `wandb login` to log in to your wandb account.** Also, please create a new project named `earnings-call-v4` on wandb.

- python 3.11
- pytorch 2.5.0 (cu124)
- lightning 2.4.0
- deepspeed 0.16.1
- hydra 1.3.2
- wandb 0.19.0
- pyarrow 16.0.0

### How to Run

#### Prepare the Data
Download all files and folders from [this link](https://1drv.ms/f/s!AqUu9ylMgqcDg4LCHRr6-KiTv4GoCOc?e=xsoL1f). Put the downloaded files and folders in a new folder `data` and place the `data` folder under the project root directory.

We provide the most recent two windows for training and testing:
- 20q3-22q2 for training and 22q3 for testing
- 20q4-22q3 for training and 22q4 for testing

The `data` folder contains the following files and folders:
- `sentence-emb-mpnet/`: The pre-computed sentence embeddings using MPNet.
- `split/split_rollqtr.feather`: A dataset that contains the training/test split information for each rolling window.
- `transcripts/dt-sents`: Contains the metadata and original text for each transcript **sentence**.
- `tx/`: Contains the metadata and structured features for each transcript.


#### Change the Configurations
Following the Hydra convention, the main code is under `gradperp/model/src/`, and the main configurations are under `gradperp/model/configs/`. The most important configuration for our proposed method is under `gradperp/model/configs/experiment/swp-proposed.yaml`.

> [!IMPORTANT] 
> The implementation of GradPerp can be found in `gradperp/model/src/weighting_methods/weighting_methods.py` (the GradPerp class) and `gradperp/model/src/models/models.py` (the FrTxtMQModel class and its ancestors).

To run the experiment, please:
- Enter the project root directory.
  - e.g., `cd path-to-reproduce-finetune`

- Change the PATH in `gradperp/model/.env`:
    - Change the `WORK_DIR` to the absolute path of the project root directory.
    - Change the `DATA_DIR` to the absolute path of the `data` folder.
    - Change the `PREEMB_DIR` to the absolute path of the `sentence-emb-mpnet` folder.
    - Change the `RUN_SCRIPT` to the absolute path of the main code.

- Run `wandb sweep the-absolute-path-to-the-experiment-config-file` to start the experiment, e.g., `wandb sweep /home/user/reproduce-finetune/gradperp/model/configs/experiment/swp-proposed.yaml`.

- By running the above line, you will register a new sweep on wandb. Wandb will generate a sweep command for you, something like `wandb agent username/earnings-call-v4/vyxrzp7k`.

- Run the sweep command (e.g., `wandb agent username/earnings-call-v4/vyxrzp7k`) to start the experiment.

- You can track the live training logs on the Wandb dashboard.

- The checkpoints will be saved in the `reproduce-finetune/gradperp/checkpoints` folder.

- The predicted results will be saved in the `reproduce-finetune/data/eval/yt-temp/(sweep-id)-[sweep-id]` folder. Since we select the most recent two windows for training and testing, we will have two result datasets, one for 22Q3 and one for 22Q4. The result dataset is stored as a feather file. You can use `pyarrow` to read it.


## Finetuning Mistral 7B and Llama 3.1 8B

### Dependencies
- python 3.11
- pytorch 2.5.0 (cu124)
- transformers 4.46.3
- datasets 3.1.0
- unsloth 2024.12.2
- polars 1.16.0

> Note 1: To speed up training and reduce memory usage, we use Unsloth for finetuning and inference. The installation of Unsloth is a bit tricky, depending on the CUDA version, pytorch version, and GPU model. Please refer to [Unsloth's documentation](https://github.com/unslothai/unsloth) for detailed instructions.

> Note 2: The code is only tested with the above dependencies. It may not work with other versions.

> Note 3: Compared to a python script, training on a notebook is about 20% slower. However, for readability and interactivity, we use a notebook in this repository.

### How to Run
#### Prepare the Data
- Download all files and folders from [this link](https://1drv.ms/f/s!AqUu9ylMgqcDg4LCHRr6-KiTv4GoCOc?e=xsoL1f). Put the downloaded files and folders in a new folder `data` and place the `data` folder under the project root directory.

#### Finetune the Model
Run `finetune.py` to finetune the model. For example, to finetune Mistral 7B on time period 20Q4-22Q3 and test on 22Q4, run:
