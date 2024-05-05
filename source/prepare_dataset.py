import glob
import re
import numpy as np
import pandas as pd


# Load dataset path
KOTLIN_DATASET_PATH = "dataset/LeetCode-Kotlin/"

# Leetcode solutions
df = pd.read_csv("dataset/LeetCode-Kotlin/leetcode_dataset - lc.csv")


def get_description(id_):
    return df.query("id == @id_")['description'].item()


WHITESPACE_PLACEHOLDER = " "

data_lst = []
for n, problem_file_path in enumerate(glob.glob(KOTLIN_DATASET_PATH+ "*/*.kt")):
    problem_difficulty = problem_file_path.split("/")[-2]
    with open(problem_file_path, "r") as reader:
        problem_id = int(problem_file_path.split("/")[-1].split(".")[0])
        description = get_description(id_=problem_id)
        
        lines = reader.readlines()
        code_processed = re.sub(r'\/\*[\s\S]*?\*\/|\/\/.*', '', " ".join(lines))
  
        regex = r"\s{2}fun(.*?)\{"
        match = re.search(regex, " ".join(lines))
        function = match.group().strip()
        signature = r"class Solution {0} {1}".format("{", function) 

        code_processed_list = " ".join([item.strip() for item in code_processed.split("\n") if item])

        body = " ".join(code_processed_list.split())


    data_lst.append({"id": problem_id, 
                     "signature": signature,
                     "docstring": description,
                     "body": body.replace(signature, "").strip(),
                     "difficulty": problem_difficulty})
    # if n == 0:
    #     break

df = pd.DataFrame(data_lst)
# print(df)

train, validate, test = np.split(df.sample(frac=1., random_state=42), [int(0.6*len(df)), int(0.8*len(df))])

train.to_json("temp/kotlin/train_dataset.json", lines=True, orient="records")
validate.to_json("temp/kotlin/valid_dataset.json", lines=True, orient="records")
test.to_json("temp/kotlin/test_dataset.json", lines=True, orient="records")
