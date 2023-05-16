
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

DEBUG = False

INPUT_DIR = 'articles'

USE_APEX = False
APEX_OPT_LEVEL = 'O1'

MODEL = 'gpt2'  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}

UNFREEZE_LAST_N = 6  # The last N layers to unfreeze for training

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

MAXLEN = 768  # {768, 1024, 1280, 1600}

if USE_APEX:
    TRAIN_BATCHSIZE = 8
    BATCH_UPDATE = 16
else:
    TRAIN_BATCHSIZE = 8
    BATCH_UPDATE = 32

EPOCHS = 10
LR = 5e-4
EPS = 1e-8
WARMUP_STEPS = 1e2

SEED = 2020
# Remove if GPU available
torch.device("cpu")

class myDataset(Dataset):

    def __init__(self, data, tokenizer, randomize=True):

        context, question, answer= [], [], []
        for k, v in data.items():
            context.append(v[0])
            question.append(v[1])
            answer.append(v[2])

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.context = context
        self.question = question
        self.answer = answer

        # ---------------------------------------------#


    def __len__(self):
        return len(self.answer)

    # ---------------------------------------------#

    def __getitem__(self, i):
        input = SPECIAL_TOKENS['bos_token'] + self.context[i] + \
                SPECIAL_TOKENS['sep_token'] + self.question[i] + SPECIAL_TOKENS['sep_token'] + \
                self.answer[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = self.tokenizer(input,
                                   truncation=True,
                                   max_length=MAXLEN,
                                   padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}

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
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    # model.cuda()
    return model

with open('test.json', 'r') as file:
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

train_dataset = myDataset(data, tokenizer)
val_dataset = myDataset(data, tokenizer, randomize=False)

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
#
tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer,
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='./content/pytorch_model.bin')

#---------------------------------------------------#



trainer.train()
trainer.save_model()
