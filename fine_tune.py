
import json
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

from data_prep import GSMDataset

import argparse

DEBUG = False

USE_APEX = False
APEX_OPT_LEVEL = 'O1'

MODEL = 'gpt2'  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}

UNFREEZE_LAST_N = 6  # The last N layers to unfreeze for training

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"
                  }

MAXLEN = 768  # {768, 1024, 1280, 1600}

if USE_APEX:
    TRAIN_BATCHSIZE = 8
    BATCH_UPDATE = 16
else:
    TRAIN_BATCHSIZE = 4
    BATCH_UPDATE = 8

EPOCHS = 10
LR = 5e-4
EPS = 1e-8
WARMUP_STEPS = 1e2

SEED = 2020

torch.device("cpu")

def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else:
        config = AutoConfig.from_pretrained(MODEL,
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)

    #----------------------------------------------------------------#
    model = GPT2LMHeadModel.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    # model.cuda() # uncomment if use cuda
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='gpt2')

    parser.add_argument('--num_steps', default = './dataset/GraspPose', type=str,
                        help='The path to the folder that contains grabpose data')
    
    train_dataset_path = "dataset/train_socratic_processed_sep.json"
    with open(train_dataset_path, 'r') as file:
        data = json.load(file)
    tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
    model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS)

    # - Freeze selective layers:
    # - Freeze all layers except last n:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        #Only un-freeze the last n transformer blocks
        if i+1 > 12 - UNFREEZE_LAST_N:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True

    train_dataset = GSMDataset(data, tokenizer)
    val_dataset = GSMDataset(data, tokenizer, randomize=False)

    training_args = TrainingArguments(
        output_dir="./content/",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCHSIZE,
        per_device_eval_batch_size=TRAIN_BATCHSIZE,
        gradient_accumulation_steps=BATCH_UPDATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        adam_epsilon=EPS,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    #---------------------------------------------------#
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()
