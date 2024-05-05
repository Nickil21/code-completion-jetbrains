import re
import json
import torch
import argparse
import pandas as pd

from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from source.model import get_prompt, reformat_input
from source.finetune import peft_fine_tune


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    tokens = [t for t in code.split(' ') if t]
    return tokens


def post_process(code):
    code = code.replace("<EOL>", "\n").replace("<INDENT>", " ").replace("<DEDENT>", " ")
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return " ".join(code.split())


def get_chat_reponse(required_length, lang, input_code):
    messages = get_prompt(lang=lang, input_code=input_code)
    inputs = tokenizer(messages, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, top_p=0.95, temperature=0.2, top_k=5, do_sample=True)
    prompt_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][prompt_length:])
    response_output = " ".join(response.strip().split())
    return response_output[:required_length]


def compute_scores(lang, task, model):
    if lang == "kotlin":
        data_kotlin = pd.read_json("temp/kotlin/test_dataset.json", orient="records", lines=True)
        data_kotlin = reformat_input(data_kotlin)

    if lang == "python":
        with open("dataset/CodeXGLUE-Python/test.json") as json_data:
            data_python = json.load(json_data)
            data_python = pd.DataFrame.from_dict(pd.DataFrame(data_python['rows'])['row'].tolist())
            data_python = reformat_input(data_python)
        
        with open("dataset/CodeXGLUE-Python/answers.txt") as file:
            test_python_answers = file.read().splitlines()

    y_true = []
    y_pred = []

    if lang == "kotlin":
        data = data_kotlin

    if lang == "python":
        data = data_python

    for i, r in data.iterrows():

        if lang == "kotlin":
            actual = r['body'].strip()

        if lang == "python":
            actual = test_python_answers[i]


        length_of_test_code = len(actual)

        print(r['input'], flush=True)
        predicted = get_chat_reponse(lang=lang, required_length=length_of_test_code, input_code=r['input'])
        
        # assert len(predicted) == len(actual), f"Samples of predictions and answers are not equal, {len(predicted)}: {len(actual)}"

        print(f"Actual -> {actual}", flush=True)
        print(f"Predicted -> {predicted}", flush=True)
        print()

        y_true.append(actual)
        y_pred.append(predicted)

        # if i == 10:
        #     break

    df_predicted = pd.DataFrame(y_pred)
    df_predicted.to_csv(f"temp/{lang}/{model.split('/')[-1]}_{task}_predictions.txt", header=None, index=None, sep="\t")

    total = len(y_true)
    edit_sim = 0.0
    bleu_score = 0.0

    for pred, gt in zip(y_pred, y_true):
        pred = post_process(pred.strip())
        gt = post_process(gt.strip())
        edit_sim += fuzz.ratio(pred, gt)

        pred_tokenized = tokenize_for_bleu_eval(pred)
        gt_tokenized = tokenize_for_bleu_eval(gt)
        bleu_score += sentence_bleu([gt_tokenized], pred_tokenized) * 100

    print(f"Model: {model}, Task: {task}, Lang: {lang}, Edit sim: {round(edit_sim/total, 2)}, BLEU: {round(bleu_score/total, 2)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint", required=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs for finetuning", default=2)
    parser.add_argument("--batch_size", type=int, help="Batch size for finetunining", default=32)
    parser.add_argument("--zero_shot_evaluate", default=False, action="store_true", help="Evaluate zero-shot performance")
    parser.add_argument("--finetune_evaluate", default=False, action="store_true", help="Evaluate finetune performance")
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                             # bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16)

    print("Loading model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
                                args.checkpoint,
                                # quantization_config=quantization_config,
                                torch_dtype=torch.float16,
                                device_map="auto"
                        )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    print("Model loaded ...", flush=True)

    if args.zero_shot_evaluate:
        compute_scores(lang="kotlin", task="zero_shot", model=args.checkpoint)
        compute_scores(lang="python", task="zero_shot", model=args.checkpoint)

    if args.finetune_evaluate:
        # Finetune
        peft_fine_tune(model_name=args.checkpoint, method="LoRA", batch_size=args.batch_size, epochs=args.epochs)

        MODEL_NAME = f"nickil/{args.checkpoint.split('/')[-1]}-LoRA-finetuned"
        config = PeftConfig.from_pretrained(MODEL_NAME)
        model = PeftModel.from_pretrained(model, MODEL_NAME)

        compute_scores(lang="kotlin", task="finetune", model=args.checkpoint)
        compute_scores(lang="python", task="finetune", model=args.checkpoint)
