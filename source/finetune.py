from datetime import datetime
import os
import sys

import torch
from peft import (
    LoraConfig,
    AdaLoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig

import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from source.model import get_finetuning_prompt, reformat_input


train_pandas = pd.read_json("temp/kotlin/train_dataset.json", lines=True, orient="records")
train_pandas = reformat_input(train_pandas)

valid_pandas = pd.read_json("temp/kotlin/valid_dataset.json", lines=True, orient="records")
valid_pandas = reformat_input(valid_pandas)

train_dataset = Dataset.from_pandas(train_pandas)
eval_dataset = Dataset.from_pandas(valid_pandas)


def peft_fine_tune(model_name, method, batch_size, epochs):

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=492, # 512 - num_virtual_tokens
            padding=False,
            return_tensors=None,
        )

        # "self-supervised learning" means the labels are also the inputs:
        result["labels"] = result["input_ids"].copy()

        return result


    def generate_and_tokenize_prompt(data_point):
        full_prompt = get_finetuning_prompt(data_point=data_point)
        return tokenize(full_prompt)

    output_dir = f"/bask/projects/j/jlxi8926-auto-sum/nmaveli/jetbrains/{model_name.split('/')[-1]}-{method}-finetuned" 

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                        # bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=quantization_config,
                                torch_dtype=torch.float16,
                                device_map="auto"
                        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    # print(tokenized_train_dataset[-1])

    model.train() # put model back into training mode
    model = prepare_model_for_kbit_training(model)

    if method == "LoRA":
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    if method == "PrefixTuning":
        config = PrefixTuningConfig(
                    num_virtual_tokens=20,
                    inference_mode=False,
                    task_type="CAUSAL_LM")

    if method == "AdaLoRA":
        config = AdaLoraConfig(
                    task_type="CAUSAL_LM", 
                    init_r=12,
                    target_r=8,
                    beta1=0.85,
                    beta2=0.85,
                    tinit=200,
                    tfinal=1000,
                    deltaT=10,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    inference_mode=False
                    )

    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    print(model)


    resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

    if resume_from_checkpoint:
        if os.path.exists(resume_from_checkpoint):
            print(f"Restarting from {resume_from_checkpoint}")
            adapters_weights = torch.load(resume_from_checkpoint)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {resume_from_checkpoint} not found")


    wandb_project = "jetbrains-code-completion"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project


    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    batch_size = batch_size
    per_device_train_batch_size = 8
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            warmup_steps=100,
            max_steps=400,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps", # if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=True, # group sequences of roughly the same length together to speed up training
            report_to="wandb", # if use_wandb else "none",
            run_name=f"{output_dir}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
            push_to_hub=True
        )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )


    model.config.use_cache = False

    old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    #     model, type(model)
    # )
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("compiling the model")
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)
    trainer.push_to_hub()
