import pandas as pd
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset

def data_preprocess(file_path, output_path, sep = False):

    df = pd.read_json(path_or_buf=file_path, lines=True)
    # Split sub-question answer pairs and generate input output pairs
    data = dict()
    for index, row in df.iterrows():

        # Remove question from the problem
        ind_q = row["question"].rfind(".") + 1
        q = row["question"][ind_q + 1:]
        context = row["question"][:ind_q]

        subq_a_pairs = row["answer"].split("\n")
        q_a_seq = []
        for i in range(len(subq_a_pairs)-1):
            subq_a = subq_a_pairs[i]
            subq, suba = subq_a.split(" ** ")
            q_a_seq.append([subq, suba])
        # the final answer
        a = subq_a_pairs[-1].split("#### ")[1]

        if sep:
            id_prefix = str(index)
            for i in range(len(q_a_seq)):
                id = id_prefix + str(i)
                data[id] = [context, q, q_a_seq[:i+1], a]
        else:
            data[index] = [context, q, q_a_seq, a]

    with open(output_path, "w") as outfile:
        json.dump(data, outfile)


    print("preprocced data have already written in {}".format(output_path))

def extract_subques_exp(dataset_path, output_path, num_steps=None):
    """ Extract specified num_steps of problems, and write into the json files
    args:
        dataset_path: the path to the dataset
        output_path: the folder to write the processed data
        num_steps: num_steps to specified, default is None (save all the steps respectively)
    """
    
    with open(dataset_path, 'r') as file:
        all_data = json.load(file)

    examples = defaultdict(list)

    for k, v in all_data.items():
        concat_ans = ""
        concat_questions = ""
        step = 0
        for sub_q, sub_a in v[2]:
            temp = {}
            step += 1
            if num_steps and step != num_steps:
                continue
            concat_questions += " " + sub_q
            concat_ans += " " + sub_a 
            temp["context"] = v[0] + " " + v[1] # context + main problem
            temp["subquestions"] = concat_questions
            temp["subanswers"] = concat_ans
            examples[step].append(temp)

    for k in examples.keys():
        output_json_path = output_path + "/train_{}_steps.json".format(str(k))
        with open(output_json_path, "w") as outfile:
            json.dump(examples[k], outfile)
        print("{} steps: {} data in total".format(k, len(examples[k])))

class GSMDataset(Dataset):
    def __init__(self, tokenizer, examples, special_tokens, loss_on_prefix=True):
        self.examples = examples
        self.context = [ex["context"] for ex in self.examples]
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.context = tokenizer(self.context, padding = False)
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.special_tokens = special_tokens
        self.max_len = max(
            [
                len(self.context["input_ids"][i]) + len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context_tokens = self.special_tokens["bos_token"] + self.context["input_ids"][idx]
        qn_tokens = self.qns["input_ids"][idx] + self.special_tokens["sep_token"]
        ans_tokens = self.ans["input_ids"][idx] + self.special_tokens["eos_token"]
        pad_tokens = [0] * (self.max_len - len(context_tokens) - len(qn_tokens) - len(ans_tokens))
        tokens = context_tokens + qn_tokens + ans_tokens + pad_tokens
        mask = (
            (([1] * len(context_tokens))
            + [int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)

class myDataset(Dataset):

    def __init__(self, data, tokenizer, randomize=True):

        context, question, sub_q_a_pairs, answer= [], [], [], []
        for k, v in data.items():
            context.append(v[0])
            question.append(v[1])
            sub_q_a_pairs.append(v[2])
            answer.append(v[3])

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.context = context
        self.question = question
        self.answer = answer
        self.subq_a_pairs = sub_q_a_pairs


    def __len__(self):
        return len(self.answer)

    def __getitem__(self, i):
        input = SPECIAL_TOKENS['bos_token'] + \
                SPECIAL_TOKENS['additional_special_tokens'][0] + self.context[i] + \
                SPECIAL_TOKENS['additional_special_tokens'][1] + self.question[i]
        
        # append the previous answers
        for j in range(len(self.subq_a_pairs[i])-1):
            input += SPECIAL_TOKENS['additional_special_tokens'][2] + self.subq_a_pairs[i][j][1]
        
        # append the last sub question
        input += SPECIAL_TOKENS['additional_special_tokens'][3] + self.subq_a_pairs[i][-1][0] + SPECIAL_TOKENS['eos_token']

        # construct the label as the last sub answer
        label = SPECIAL_TOKENS['bos_token'] + self.subq_a_pairs[i][-1][1] + SPECIAL_TOKENS['eos_token']

        encodings_dict_input = self.tokenizer(input,
                                   truncation=True,
                                   max_length=MAXLEN,
                                   padding="max_length")

        encodings_dict_label = self.tokenizer(label,
                                   truncation=True,
                                   max_length=MAXLEN,
                                   padding="max_length")
        
        input_ids = encodings_dict_input['input_ids']
        attention_mask = encodings_dict_input['attention_mask']
        label_ids = encodings_dict_label['input_ids']

        return {'label': torch.tensor(label_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}


if __name__ == '__main__':
    dataset_path = "dataset/train_socratic_processed_sep.json"
    output_path = "dataset"
    extract_subques_exp(dataset_path, output_path, num_steps=None)
