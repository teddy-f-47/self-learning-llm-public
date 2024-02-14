from transformers import AutoTokenizer, AutoConfig
import lightning as pl
import pickle
import torch
import wandb
import os

from self_learning_training import do_train


wandb.login()

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
pl.seed_everything(47, workers=True)

torch.set_float32_matmul_precision("high")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"]="self_learning_training"

pretrained_model_name = "Intel/neural-chat-7b-v3-3"
ds_filepath = "results/ds_Intel_neural-chat-7b-v3-3_open.pickle"
result_filepath = "results/open_generation/res_Intel_neural-chat-7b-v3-3_OpenGen.pickle"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    use_fast=True
)
model_config = AutoConfig.from_pretrained(pretrained_model_name)
model_vocab_size = model_config.vocab_size
print(f"model_config.vocab_size: {model_config.vocab_size}")
print(f"default len(tokenizer): {len(tokenizer)}")

if model_vocab_size > len(tokenizer):
    model_vocab_size_diff = model_vocab_size - len(tokenizer)
    additional_special_tokens = [f"<pad{token_id}>" for token_id in range(model_vocab_size_diff)]
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    print(f"extended len(tokenizer): {len(tokenizer)}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
print(f"tokenizer.eos_token: {tokenizer.eos_token}")

def prompt_fn(prompt):
    return f"### System:\nYou are a student who is eager to learn about new things.\n### User:\n{prompt}\n### Assistant:\n"

def extract_response_fn(response):
    return response.split("### Assistant:\n")[-1]

with open(ds_filepath, 'rb') as dump_handle:
    ds = pickle.load(dump_handle)

with open(result_filepath, 'rb') as dump_handle:
    result = pickle.load(dump_handle)
questions_with_hallucination = result["prompts_with_hallucination"]

ckpt_filepath = do_train(
    ds=ds,
    questions_with_hallucination=questions_with_hallucination,
    prompt_format_fn=prompt_fn,
    extract_response_fn=extract_response_fn,
    tokenizer=tokenizer,
    model_name_or_path=pretrained_model_name,
    batch_size=4,
    max_epochs=10,
    lr=3e-5,
    deterministic=True
)
print(f"ckpt_filepath: {ckpt_filepath}")
