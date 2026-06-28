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
import argparse
from functools import partial
import os
os.environ["WANDB_DISABLED"] = "true"

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
                if conversations[i]['role'] == 'assistant':
                    cur_input_str = "<|im_start|>assistant\n" + conversations[i]['content'] + "<|im_end|>\n"
                    cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
                    labels.extend([-100] * len(cur_input_ids['input_ids']))
                else:
                    cur_input_str = "<|im_start|>memory_query\n" + conversations[i]['content'] + "<|im_end|>\n"
                    cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
                    labels.extend([-100] * len(cur_input_ids['input_ids']))
            else:
                cur_input_str = "<|im_start|>assistant\n"
                label_str = conversations[i]['content'] + tokenizer.eos_token
                cur_input_ids = tokenizer(cur_input_str + label_str, add_special_tokens=False)
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
                label_str = "<think>" + conversations[i]['think'] + " </think>" + conversations[i][
                    'content'] + tokenizer.eos_token
                cur_input_str = "<|im_start|>assistant\n"
                cur_input_ids = tokenizer(cur_input_str + label_str, add_special_tokens=False)
                labels.extend([-100] * len(tokenizer(cur_input_str, add_special_tokens=False)['input_ids']))
                labels.extend(tokenizer(label_str, add_special_tokens=False)['input_ids'])
        input_ids.extend(cur_input_ids['input_ids'])
        attention_mask.extend(cur_input_ids['attention_mask'])
        input_str += cur_input_str
    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def process_func_conversations(example, tokenizer):
    messages = example['conversations']
    input_str = f"<|im_start|>system\nYou are an AI assistant. When chatting with the user, if the user mentions something from the past, you need to recall by calling the function memory_query_call and passing in the retrieval query. The content of the query must include a part that describes the time and a part that contains key semantic information. Then, based on the memory returned by the memory_query role, think about the memory fragments within the `<think></think>` block, and generate the correct response based on the sorted information. If the user does not mention anything from the past, generate the correct response according to the current context.\n"
    input_str_ids = tokenizer(input_str, add_special_tokens=False)
    input_ids = []
    input_ids.extend(input_str_ids["input_ids"])
    attention_mask = []
    attention_mask.extend(input_str_ids["attention_mask"])
    labels = []
    labels.extend([-100] * len(input_str_ids["input_ids"]))
    for i in range(len(messages)):
        # pass
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend([-100] * len(cur_input_ids['input_ids']))
        else:
            cur_input_str = ""
            if 'think' in messages[i] and messages[i]['think'] is not None:
                cur_input_str += "<think>" + messages[i]['think'] + "</think>"
            cur_input_str += messages[i]['content'] + tokenizer.eos_token
            cur_input_ids = tokenizer(cur_input_str, add_special_tokens=False)
            input_ids.extend(cur_input_ids['input_ids'])
            attention_mask.extend(cur_input_ids['attention_mask'])
            labels.extend(cur_input_ids['input_ids'])
        input_str += cur_input_str
    input_ids = (input_ids)
    attention_mask = attention_mask
    labels = (labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length": len(input_ids)}

def filter_by_length(example):
    """过滤掉长度大于4096的样本"""
    return example["length"] <= 4096

def predict_conversation(example, model, tokenizer):
    messages = example['conversations']
    whole_str = f"<|im_start|>system\nYou are an AI assistant. When chatting with the user, if the user mentions something from the past, you need to recall by calling the function memory_query_call and passing in the retrieval query. The content of the query must include a part that describes the time and a part that contains key semantic information. Then, based on the memory returned by the memory_query role, think about the memory fragments within the `<think></think>` block, and generate the correct response based on the sorted information. If the user does not mention anything from the past, generate the correct response according to the current context.\n"
    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            cur_input_str = "<|im_end|>\n<|im_start|>user\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs,
                                    max_new_tokens=384,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,)
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response
        elif messages[i]['role'] == 'memory_query':
            cur_input_str = "<|im_end|>\n<|im_start|>memory_query\n" + messages[i]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            whole_str += cur_input_str
            inputs = tokenizer(whole_str, add_special_tokens=False, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs,
                                     max_new_tokens=384,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id, )
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            whole_str += response
    print("whole_str:", whole_str)
    print("----------------------")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--train_trigger_json_path", type=str, required=True,
                        help="Path to the training JSON file")
    parser.add_argument("--train_reasoning_json_path", type=str, required=True,
                        help="Path to the training JSON file")
    parser.add_argument("--train_conversations_json_path", type=str, required=True,
                        help="Path to the training JSON file")

    parser.add_argument("--test_conversations_json_path", type=str, required=True,
                        help="Path to the test JSON file")
    parser.add_argument("--lora_path",  type=str, required=True, 
                        help="Lora Path after trigger and reasonging training")

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
    train_conversations_json_path = args.train_conversations_json_path
    test_conversations_json_path = args.test_conversations_json_path

    model_dir = args.model_dir
    lora_path = args.lora_path
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

    # 少量触发数据
    train_trigger_df = pd.read_json(train_trigger_json_path)
    train_trigger_dataset = Dataset.from_pandas(train_trigger_df)
    process_func_trigger = partial(process_func_trigger, tokenizer=tokenizer)
    train_trigger_dataset = train_trigger_dataset.map(process_func_trigger, num_proc=1, remove_columns=train_trigger_dataset.column_names)
    train_trigger_dataset = train_trigger_dataset.filter(filter_by_length, num_proc=1)
    train_trigger_dataset = train_trigger_dataset.shuffle(seed=42)
    trigger_subset = train_trigger_dataset.select(range(int(len(train_trigger_dataset) * 0.05)))

    # 少量推理数据
    train_reasoning_df = pd.read_json(train_reasoning_json_path)
    train_reasoning_dataset = Dataset.from_pandas(train_reasoning_df)
    process_func_reasoning = partial(process_func_reasoning, tokenizer=tokenizer)
    train_reasoning_dataset = train_reasoning_dataset.map(process_func_reasoning, num_proc=1, remove_columns=train_reasoning_dataset.column_names)
    train_reasoning_dataset = train_reasoning_dataset.filter(filter_by_length, num_proc=1)
    train_reasoning_dataset = train_reasoning_dataset.shuffle(seed=42)
    reasoning_subset = train_reasoning_dataset.select(range(int(len(train_reasoning_dataset) * 0.05)))

    # 多轮对话数据
    train_conversations_df = pd.read_json(train_conversations_json_path)
    train_conversations_dataset = Dataset.from_pandas(train_conversations_df)
    process_func_conversations = partial(process_func_conversations, tokenizer=tokenizer)
    train_conversations_dataset = train_conversations_dataset.map(process_func_conversations, num_proc=1, remove_columns=train_conversations_dataset.column_names)
    train_conversations_dataset = train_conversations_dataset.filter(filter_by_length, num_proc=1)

    train_dataset = concatenate_datasets([trigger_subset, reasoning_subset, train_conversations_dataset])
    train_dataset = train_dataset.shuffle()

    test_conversations_df = pd.read_json(test_conversations_json_path)
    test_conversations_dataset = Dataset.from_pandas(test_conversations_df)

    lora_model_path = lora_path
    model = PeftModel.from_pretrained(model, lora_model_path, is_trainable=True)
    print("load OK")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,  # 适当降低学习率，从 2e-5 降到 1.5e-5，提高稳定性
        save_on_each_node=True,
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("begin train")
    trainer.train()

    model.eval()
    print("begin test")
    test_samples = len(test_conversations_dataset)
    for i in range(test_samples):
        example = test_conversations_dataset[i]
        predict_conversation(example, model, tokenizer)
    print("test end")