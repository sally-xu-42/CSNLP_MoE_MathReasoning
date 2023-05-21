import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from data_prep import GSMDataset, split_data
import wandb
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import EarlyStopping


@hydra.main(config_path='configs', config_name='train')
def main(args: DictConfig):
    # initialize logging
    log_directory = os.getcwd()

    print(f'log_directory: {log_directory}')
    wandb_name = f'{args.model}'
    wandb_name += f' -num_steps {args.num_steps}' if args.num_steps else ''
    wandb.init(project='MOE-MWP', name=wandb_name, notes='', dir=log_directory,
               settings=wandb.Settings(start_method='fork'), mode=args.wandb_mode)
    args_to_log = dict(args)
    args_to_log['out_dir'] = log_directory
    ckpt_path = args.checkpoint_dir
    ckpt_path += args.model
    ckpt_path += f'/seed{args.seed}'
    ckpt_path += f'/epochs{args.num_epochs}'
    ckpt_path += f'/steps_n_{args.num_steps}' if args.num_steps else ''
    args_to_log['ckpt_path'] = ckpt_path
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    wandb.config.update(args_to_log)
    del args_to_log
    os.makedirs(ckpt_path, exist_ok=True)

    # load dataset
    train_dataset_path = args.repo_dir + f'/{args.dataset_path}'
    train_dataset_path += f'/train_{args.num_steps}_steps.json' if args.num_steps else f'/train_all_steps.json'
    with open(train_dataset_path, 'r') as file:
        data = json.load(file)
    train_examples, val_examples = split_data(data, split_ratio=0.9)

    # TODO think about how to set special tokens, whether ot not need to add additional ones
    SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                      "eos_token": "<|EOS|>",
                      "unk_token": "<|UNK|>",
                      "pad_token": "<|PAD|>",
                      "sep_token": "<|SEP|>"
                     }
    
    # set up tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    
    # add special tokens
    if SPECIAL_TOKENS:
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        print("Special tokens added")
    
    device = torch.device(args.device)

    train_dset = GSMDataset(tokenizer, train_examples, SPECIAL_TOKENS, 768, device)
    valid_dset = GSMDataset(tokenizer, val_examples, SPECIAL_TOKENS, 768, device)

    print("Load data with {} steps successfully!".format(args.num_steps if args.num_steps else "all"))
    print("Train data set size: {}".format(len(train_dset)))
    print("Validation data set size: {}".format(len(valid_dset)))
    print('=====================')

    

    # load gpt2 pretrained models
    model = GPT2LMHeadModel.from_pretrained(args.model)

    # if special tokens added, model needs to be resized accordingly
    if SPECIAL_TOKENS:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    # Hyperparameters
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=args.learning_rate)

    batch_update = args.batch_update

    num_training_steps = (args.num_epochs * len(train_loader)) // batch_update

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    early_stopping = EarlyStopping(ckpt_path = ckpt_path, patience=args.patience, verbose=False, delta=args.delta)

    pbar = tqdm(range(num_training_steps))
    for epoch in range(args.num_epochs):
        for i, batch in enumerate(train_loader):
            model.train()
            # batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch, labels=batch["input_ids"])
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            if (i+1) % batch_update == 0:
                optim.step()
                lr_scheduler.step()
                pbar.update(1)
                pbar.set_description(f"train_loss: {loss.item():.5f}")
                wandb.log({"Train_loss": loss.item()})
                optim.zero_grad()

        if epoch % int(args.validation_epochs) == 0:
            total_loss, batch_count = 0, 0
            model.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    # batch = {k: v.to(device) for k, v in batch.items()}
                    # outputs = model(**batch, labels=batch["input_ids"])
                    outputs = model(**batch)
                    loss = outputs[0]
                    batch_count+=1
                    total_loss +=loss.item()
            valid_loss = total_loss/batch_count
            wandb.log({"Valid_loss": valid_loss, "Epochs": epoch})
            
            # for early stop
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

if __name__ == "__main__":
    main()
