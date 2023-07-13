from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import csv
import os
import sys
import time
from calculator import sample

num_steps = sys.argv[1]

curr_dir = "/cluster/home/daizhang/MixtureOfExpertMathReasoning/"
ckpt_pth = curr_dir + "checkpoints/steps_n_{}".format(str(num_steps))
data_pth = curr_dir + "dataset_test2/test_{}_steps.json".format(str(num_steps))

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}
tokenizer.add_special_tokens(SPECIAL_TOKENS)

# model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

model = AutoModelWithLMHead.from_pretrained(ckpt_pth)
model.to("cuda")

with open(data_pth, 'r') as file:
  all_data = json.load(file)

### Start evaluation
print("Start evaluation for num_steps={}".format(num_steps))
print("Length={}".format(len(all_data)))
start_time = time.time()

with open(f"{ckpt_pth}/test_pred2.csv", "w") as pred_file:
    correct_count = 0
    write = csv.writer(pred_file)
    for count, text in enumerate(all_data):
        qa_pairs = text['qa-pairs']
        input = SPECIAL_TOKENS['bos_token'] + text['context'] + \
            '' + text['main-q'] + SPECIAL_TOKENS['sep_token'] + ' '

        for i in range(len(qa_pairs)):
            if i == len(qa_pairs)-1:
                input += (qa_pairs[i][0] + ' ')
            else:
                input += (qa_pairs[i][0] + ' ' + qa_pairs[i][1] + SPECIAL_TOKENS['sep_token'] + ' ')

        output = qa_pairs[i][1]

        # Extract number from ground truth answer
        try: gt = float(text['answer'])
        except: gt = None

        # Prediction
        gen_text, ans = sample(model, input, tokenizer, "cuda", 256)
        try: ans = float(ans)
        except: ans = None

        if gt == ans and gt != None:
            correct_count += 1
            write.writerow([str(count) + ". LLM generated: " + gen_text])
            write.writerow([str(count) + ". Ground Truth: " + output])

    write.writerow(["Accuracy: " + str(correct_count / len(all_data))])

end_time = time.time()
elapsed_time = end_time - start_time

print("Time taken: ", elapsed_time)
