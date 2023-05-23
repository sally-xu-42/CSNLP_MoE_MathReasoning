import os
import torch as th
from calculator import sample
import json
from tqdm.auto import tqdm
import csv
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, GPTNeoForCausalLM, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import *
# import wandb


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def calculate_acc(prediction, gt):
    pred_ans = prediction
    parts = gt.split(">>")
    if len(parts) > 0:
        try:
            gt = float(parts[-1].split(" ")[0])
        except:
            return int(0)
    else:
        return int(0)

    # gt = extract_answer(gt)

    if pred_ans == gt:
        return int(1)
    else:
        return int(0)

def write_to_csv(model_ckpt_path, in_seq, out_seq, predicted_out, accuracy):

    """Write input, ground truth and prediction to the CSV file
    """

    with open(f"{model_ckpt_path}/test_pred_gsm.csv", "w") as pred_file:
        write = csv.writer(pred_file)
        write.writerow(["Question", "Ground Truth", "Prediction", "Accuracy"])
        for in_samp, out_samp, pred_samp, acc in zip(in_seq, out_seq, predicted_out, accuracy):
            write.writerow([in_samp, out_samp, pred_samp, acc])
        print(f"File saved at :{model_ckpt_path}")

@hydra.main(config_path='configs', config_name='test_gsm')
def main(args: DictConfig):
    # initialize logging
    # log_directory = os.getcwd()
    # print(f'log_directory: {log_directory}')
    # wandb_name = f'{args.model}'
    # wandb_name += f'standard-training' if 'step' not in args.checkpoint_path else ''
    # wandb_name += f' -d gsm'
    # wandb_name += f' -num_steps {args.num_steps}' if args.num_steps else ''
    # #wandb_name += f' {"subq" if args.subquestions else "no-subq"}'
    # wandb.init(project='distill-MWP-eval', name=wandb_name, notes='', dir=log_directory,
    #            settings=wandb.Settings(start_method='fork'), mode=args.wandb_mode)
    # args_to_log = dict(args)
    # args_to_log['out_dir'] = log_directory
    # print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    # wandb.config.update(args_to_log)
    # del args_to_log

    device = th.device(args.device)
    model_ckpt_path = args.checkpoint_path

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    # add special tokens
    if SPECIAL_TOKENS:
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        print("Special tokens added")

    model_class = (GPT2LMHeadModel if 'gpt2' == args.model else
                   GPTNeoForCausalLM if 'gpt-neo' == args.model else
                   AutoModelForCausalLM if 'gpt-j' == args.model else
                   None)
    model = model_class.from_pretrained(model_ckpt_path)

    model.to(device)
    print("Model Loaded")

    with open(args.test_data_path) as fh:
        all_data = [json.loads(line) for line in fh.readlines() if line]
    input_sequences = []
    out_sequences = []
    for datum in all_data[0]:
        input_sequences.append(datum["context"] + SPECIAL_TOKENS["sep_token"] + datum["subquestions"] + SPECIAL_TOKENS["sep_token"])
        out_sequences.append(datum['subanswers'])

    predicted_out = []
    pbar = tqdm(range(len(input_sequences)))
    sample_len = 256
    overall_accuracy = []
    for in_sample, out_sample in tqdm(zip(input_sequences, out_sequences)):
        decoded_out, answer = sample(model, in_sample, tokenizer, device, sample_len)
        predicted_out.append(''.join(decoded_out))
        overall_accuracy.append(calculate_acc(answer, out_sample))
        pbar.update(1)

    write_to_csv(model_ckpt_path, input_sequences, out_sequences, predicted_out, overall_accuracy)

    final_acc = sum(overall_accuracy)/len(overall_accuracy)

    print(f"Final Accuracy: {final_acc}")
    # wandb.run.summary.update({'final_accuracy': final_acc})


if __name__ == "__main__":
    main()
