import pandas as pd
import json

file_path = "train_socratic.jsonl"
df = pd.read_json(path_or_buf=file_path, lines=True)

# Split sub-question answer pairs and generate input output pairs
data = dict()
for index, row in df.iterrows():

    # Remove question from the problem
    ind_q = row["question"].rfind(".") + 1
    q = row["question"][ind_q + 1:]
    context = row["question"][:ind_q]

    subq_a_pairs = row["answer"].split("\n")
    id_prefix = str(index)
    for i in range(len(subq_a_pairs) - 1):
        subq_a = subq_a_pairs[i]
        subq, suba = subq_a.split(" ** ")
        id = id_prefix + "_" + str(i)
        data[id] = [context, subq, suba]

        # Append the previous sub-question and answers
        context = " ".join([context, subq, suba])
    # Append final answer
    i = len(subq_a_pairs) - 1
    id = id_prefix + "_" + str(i)
    a = subq_a_pairs[i].split("#### ")[1]
    data[id] = [context, q, a]

with open("reconstructed_qa_pairs.json", "w") as outfile:
    json.dump(data, outfile)


print("end")
