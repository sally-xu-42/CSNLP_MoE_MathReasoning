# Applying Mixture of Experts Technique on Small-Size Language Models for Multi-Step Mathematical Reasoning Problems
## Group 12 
### Team member
* Zhiyi Chen (zhiychen)
* Yaqi Qin (yaqqin)
* Tianyang Xu (tianyxu)
* Daiwei Zhang (daizhang)

This repository contains the code and preprocessed data we used for project of the course CSNLP.

## Setting up on Euler
```bash
git clone https://github.com/sally-xu-42/CSNLP_MoE_MathReasoning.git
cd CSNLP_MoE_MathReasoning
mkdir csnlp_venv
python3 -m venv csnlp_venv
source csnlp_venv/bin/activate
module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
pip install -r requirements.txt
wandb login
```

## Dataset
The original data is adapted from the Socratic version of the [GSM8K](https://github.com/openai/grade-school-math) dataset. Methods in `data_prep.py` are used to preprocessed and decomposed the training and testing datasets by the number of reasoning steps. We have two data formats: all preprocessed data in the first format is in the `dataset` directory, where each problem is in the form of, for example
```
{"context": "c. main-q.", "subquestions": "q1? q2? q3?", "subanswers": "a1. a2. a3"}
```
data in the other format is in `dataset_dialog` directory, which is formed as
```
{"context": "c", "main-q": "q", "qa-pairs":[["q1?", "a1."], ["q2?", "a2."], ...], "answer": "a"}
```
Note that all equations in the sub-answers are in form as `a+b=<<a+b=c>>c` for convenient evaluation

## Training
Config file for `train.py` is in `configs/train.yaml`. Set `repo_dir` to your working directory; set `checkpoint_dir` to `"checkpoints/"` if you want to save the trained model under the working directory, otherwise specify. 

If you want run experiments with [GPT-2](https://huggingface.co/gpt2) with 124M parameters, make sure 
```
model: "gpt2"
dataset_path: "dataset"
```
If experimenting with [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small?text=Hi.), make
```
model: "microsoft/DialoGPT-small"
dataset_path: "dataset_dialog"
```
and most importantly, modify `num_steps` in `train.yaml` to a integer number from 2 to 8 to train a specific expert model or None to train a baseline mode.

If run locally, simply `python train.py`; otherwise submit a job to the batch system  with `sbatch jobs/train` after updating your python venv path in the script.

## Inference
For GPT-2 inference, simply modify `checkpoint_path` and `test_data_path` to the trained expert model and its corresponding dataset and run `python eval_gpt2.py`. The output will be saved in a csv file and the final accuracy will be reported. 

For DialoGPT GT inference described in the report, modify the three parameters `curr_dir`, `ckpt_pth`, `data_pth` to your own paths and run `python eval_dialogpt.py n` where `n` represents the number of steps you want to evaluate.

For DialoGPT's iterative inference, please refer to this [Colab notebook](https://colab.research.google.com/drive/1rOKXyNm_6mMfeMOV6nPeugQ0Bwj8Tsjk?usp=sharing) for interactive illustration.
