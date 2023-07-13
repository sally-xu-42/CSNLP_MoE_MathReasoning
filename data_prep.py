import pandas as pd
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import random

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


def extract_subques_exp(dataset_path, output_path, mode="train", num_steps=None):
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
            temp["context"] = v[0] + " " + v[1]  # context + main problem
            temp["subquestions"] = concat_questions
            temp["subanswers"] = concat_ans
            examples[step].append(temp)

    for k in examples.keys():
        output_json_path = output_path + "/{}_{}_steps.json".format(mode, str(k))
        with open(output_json_path, "w") as outfile:
            json.dump(examples[k], outfile)
        print("{} steps: {} data in total".format(k, len(examples[k])))

def extract_subques_exp_test(dataset_path, output_path, mode = "train", num_steps=None):
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
        if num_steps and len(v[2]) != num_steps:
            continue
        concat_ans = ""
        concat_questions = ""
        # step = 0
        for sub_q, sub_a in v[2]:
            temp = {}
            # step += 1
            concat_questions += " " + sub_q
            concat_ans += " " + sub_a 
        temp["context"] = v[0] + " " + v[1] # context + main problem
        temp["subquestions"] = concat_questions
        temp["subanswers"] = concat_ans
        examples[len(v[2])].append(temp)

    for k in examples.keys():
        output_json_path = output_path + "/{}_{}_steps.json".format(mode, str(k))
        with open(output_json_path, "w") as outfile:
            json.dump(examples[k], outfile)
        print("{} steps: {} data in total".format(k, len(examples[k])))

def extract_all_exp(dataset_path, output_path, mode = "train"):
    """ Extract all num_steps of problems, and write into the json files
    args:
        dataset_path: the path to the dataset
        output_path: the folder to write the processed data
    """
    
    with open(dataset_path, 'r') as file:
        all_data = json.load(file)

    examples = []

    for k, v in all_data.items():
        concat_ans = ""
        concat_questions = ""
        for sub_q, sub_a in v[2]:
            temp = {}
            concat_questions += " " + sub_q
            concat_ans += " " + sub_a 
            temp["context"] = v[0] + " " + v[1] # context + main problem
            temp["subquestions"] = concat_questions
            temp["subanswers"] = concat_ans
            examples.append(temp)

    output_json_path = output_path + f"/{mode}_all_steps.json"
    with open(output_json_path, "w") as outfile:
        json.dump(examples, outfile)
    print("{} data in total".format(len(examples)))


def extract_all_exp_test(dataset_path, output_path, mode="train"):
    """ Extract all num_steps of problems, and write into the json files
    args:
        dataset_path: the path to the dataset
        output_path: the folder to write the processed data
    """

    with open(dataset_path, 'r') as file:
        all_data = json.load(file)

    examples = []

    for k, v in all_data.items():
        concat_ans = ""
        concat_questions = ""
        for sub_q, sub_a in v[2]:
            temp = {}
            concat_questions += " " + sub_q
            concat_ans += " " + sub_a
        temp["context"] = v[0] + " " + v[1]  # context + main problem
        temp["subquestions"] = concat_questions
        temp["subanswers"] = concat_ans
        examples.append(temp)

    output_json_path = output_path + f"/{mode}_all_steps.json"
    with open(output_json_path, "w") as outfile:
        json.dump(examples, outfile)
    print("{} data in total".format(len(examples)))

def split_data(data, split_ratio=0.9):

    # Split into training and validation sets    
    train_size = int(split_ratio * len(data))

    return data[:train_size], data[train_size:]

class GSMDatasetTA(Dataset):
    def __init__(self, tokenizer, examples, special_tokens, loss_on_prefix=True):
        """Construct the input as <bos> context + main_q + all_subq <sep> all_suba <eos>
        """
        self.examples = examples
        self.context = [special_tokens["bos_token"] + ex["context"] for ex in self.examples]
        self.qns = [ex["subquestions"] + special_tokens["sep_token"] for ex in self.examples]
        self.ans = [ex["subanswers"] + special_tokens["eos_token"] for ex in self.examples]
        self.context = tokenizer(self.context, padding = False)
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
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
        context_tokens = self.context["input_ids"][idx]
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
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

class GSMDataset(Dataset):
    def __init__(self, tokenizer, examples, special_tokens, max_len, device):
        """Construct the input as dialog style: <bos> context + main_q + <seq> + \
        subq1 + suba1 + <seq> + subq2 + suba2 + <seq> + ...
        + <eos> for dialogpt
        """
        context, qns, ans = [], [], []
        for ex in examples:
            context.append(ex["context"])
            qns.append(ex["subquestions"])
            ans.append(ex["subanswers"])
        self.context = context
        self.qns = qns
        self.ans = ans
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, i):
        input = self.special_tokens['bos_token'] + self.context[i] + \
                self.special_tokens['sep_token'] + self.qns[i] + \
                self.ans[i] + self.special_tokens['eos_token']
        
        encodings_dict_input = self.tokenizer(input,
                                   truncation=True,
                                   max_length=self.max_len,
                                   padding="max_length")
        input_ids = torch.tensor(encodings_dict_input['input_ids'])
        attention_mask = torch.tensor(encodings_dict_input['attention_mask'])
        return {'labels': input_ids.to(self.device),
                'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device)}
    
class GSMDatasetDialog(Dataset):
    def __init__(self, tokenizer, examples, special_tokens, max_len, device):
        """Construct the input as <bos> context + main_q + all_subq <sep> all_suba <eos>
        """
        context, main_q, qa_pairs = [], [], []
        for ex in examples:
            context.append(ex["context"])
            main_q.append(ex["main-q"])
            qa_pairs.append(ex["qa-pairs"])
        self.context = context
        self.main_q = main_q
        self.qa_pairs = qa_pairs

        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_len = max_len
        self.device = device

    def __len__(self):
        assert len(self.context) == len(self.main_q)
        assert len(self.main_q) == len(self.qa_pairs)
        return len(self.context)

    def __getitem__(self, i):
        
        # assert len(questions) == len(answers)
        input = self.special_tokens['bos_token'] + self.context[i] + \
            ' ' + self.main_q[i] + self.special_tokens['sep_token'] + ' '
        
        for j in range(len(self.qa_pairs[i])):
            input += (self.qa_pairs[i][j][0] + ' ' + self.qa_pairs[i][j][1])
            if j != len(self.qa_pairs[i])-1:
                input += (self.special_tokens['sep_token'] + ' ')
            else:
                input += self.special_tokens['eos_token']
        
        encodings_dict_input = self.tokenizer(input,
                                   truncation=True,
                                   max_length=self.max_len,
                                   padding="max_length")
        input_ids = torch.tensor(encodings_dict_input['input_ids'])
        attention_mask = torch.tensor(encodings_dict_input['attention_mask'])
        return {'labels': input_ids.to(self.device),
                'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device)}


if __name__ == '__main__':

    # get training dataset
    # train_dataset_path = "dataset/train_socratic_processed_sep.json"
    # train_preprocessed_path = "dataset/train_socratic_processed_sep.json"
    # data_preprocess(train_dataset_path, train_preprocessed_path, sep = False)
    # extract_subques_exp(train_preprocessed_path, "dataset", mode = "train", num_steps=None)
    # extract_all_exp(train_preprocessed_path, "dataset", mode = "train")

    # get testing dataset
    test_dataset_path = "dataset/test_socratic.jsonl"
    test_preprocessed_path = "dataset/test_socratic_processed_sep.json"
    data_preprocess(test_dataset_path, test_preprocessed_path, sep = False)
    extract_subques_exp_test(test_preprocessed_path, "dataset", mode = "test", num_steps=None)
    extract_all_exp_test(test_preprocessed_path, "dataset", mode = "test")
