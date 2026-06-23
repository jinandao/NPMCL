import json
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
from functools import partial

def process_func_trigger(example, tokenizer):
    conversations = example['conversation']
    input_str = f"<|im_start|>system\nYou are an AI assistant who is good at identifying moments that require recalling information during conversations. For example, when you encounter words referring to the past, such as yesterday, the day before yesterday, before, last time, etc., you will correctly invoke memory_query_call, then generate the correct content that needs to be recalled. The content should include a part describing time and a part with key semantic information, and finally fill the content in <content></content>.\n"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        else:
            if i != len(conversations) - 1:
                cur_input_str = "<|im_start|>assistant\n" + conversations[i]['content'] + "<|im_end|>\n"
                cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
                labels.extend([-100] * len(cur_input_ids['input_ids']))
            else:
                cur_input_str = "<|im_start|>assistant\n"
                label_str = conversations[i]['content'] + tokenizer.eos_token
                cur_input_ids = tokenizer(cur_input_str+label_str, add_special_tokens=False)
                labels.extend([-100] * len(tokenizer(cur_input_str, add_special_tokens=False)['input_ids']))
                labels.extend(tokenizer(label_str, add_special_tokens=False)['input_ids'])
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        input_str += cur_input_str
    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def process_func_reasoning(example, tokenizer):
    conversations = example['conversations']
    input_str = f"<|im_start|>system\nYou are an AI assistant skilled at making correct inferences based on memory during conversations. Your inferences must be strictly based on memory data. You will first perform the inference in <think></think>, and then output the conversation after the inference is complete.\n"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(conversations)):
        if conversations[i]['role'] == 'user':
            cur_input_str = "<|im_start|>user\n" + conversations[i]['content'] + "<|im_end|>\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        else:
            if i != len(conversations) - 1:
                if conversations[i]['role'] == 'assistant':
                    cur_input_str = "<|im_start|>assistant\n" + conversations[i]['content'] + "<|im_end|>\n"
                    cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
                    labels.extend([-100] * len(cur_input_ids['input_ids']))
                else:
                    cur_input_str = "<|im_start|>memory_query\n" + conversations[i]['content'] + "<|im_end|>\n"
                    cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
                    labels.extend([-100] * len(cur_input_ids['input_ids']))
            else:
                label_str = "<think>" + conversations[i]['think'] + " </think>"+conversations[i]['content'] + tokenizer.eos_token
                cur_input_str = "<|im_start|>assistant\n"
                cur_input_ids = tokenizer(cur_input_str+label_str, add_special_tokens=False)
                labels.extend([-100] * len(tokenizer(cur_input_str, add_special_tokens=False)['input_ids']))
                labels.extend(tokenizer(label_str, add_special_tokens=False)['input_ids'])
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        input_str += cur_input_str
    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    """过滤掉长度大于4096的样本"""
    return example["length"] <= 4096

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Query SFT Training Script")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--train_trigger_json_path", type=str, required=True,
                        help="Path to the training JSON file")
    parser.add_argument("--train_reasoning_json_path", type=str, required=True,
                        help="Path to the training JSON file")

    parser.add_argument("--output_dir", type=str, default="./Output/Memory_Query",
                        help="Model output directory")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_trigger_json_path = args.train_trigger_json_path
    train_reasoning_json_path = args.train_reasoning_json_path

    model_dir = args.model_dir
    output_dir = args.output_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    gradient_checkpointing = args.gradient_checkpointing

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 启用8bit量化
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )

    train_trigger_df = pd.read_json(train_trigger_json_path)
    train_trigger_dataset = Dataset.from_pandas(train_trigger_df)
    process_func_trigger = partial(process_func_trigger, tokenizer=tokenizer)
    train_trigger_dataset = train_trigger_dataset.map(process_func_trigger, num_proc=1)
    train_trigger_dataset = train_trigger_dataset.filter(filter_by_length, num_proc=1)

    train_reasoning_df = pd.read_json(train_reasoning_json_path)
    train_reasoning_dataset = Dataset.from_pandas(train_reasoning_df)
    process_func_reasoning = partial(process_func_reasoning, tokenizer=tokenizer)
    train_reasoning_dataset = train_reasoning_dataset.map(process_func_reasoning, num_proc=1)
    train_reasoning_dataset = train_reasoning_dataset.filter(filter_by_length, num_proc=1)

    train_dataset = concatenate_datasets([train_trigger_dataset, train_reasoning_dataset])
    train_dataset = train_dataset.shuffle()

    config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1,  # Dropout 比例
        )
    model = get_peft_model(model, config)
    print("load OK")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,  # 适当降低学习率，从 2e-5 降到 1.5e-5，提高稳定性
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("begin train")
    trainer.train()

    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)
