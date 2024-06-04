from transformers import AutoTokenizer
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

pretrained_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
ds_filepath = "results/train_ds_mistralai_Mistral-7B-Instruct-v0_2_OracleSelected.pickle"
result_filepath = "results/oracle_selected/res_mistralai_Mistral-7B-Instruct-v0_2_OracleSelected.pickle"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    use_fast=True,
    token=os.getenv('hf_personal_access_token')
)

def prompt_fn(prompt):
    text = "<s>[INST] You are a student who is eager to learn about new things. [/INST]"
    text = text + "I am a student who is eager to learn about new things. I am aware of my lack of knowledge about some things.</s> "
    return text + f"[INST] {prompt} [/INST]"

def extract_response_fn(response):
    return response.split(" [/INST]")[-1]

with open(result_filepath, 'rb') as dump_handle:
    result = pickle.load(dump_handle)
questions_with_hallucination = result["prompts_with_hallucination"]

with open(ds_filepath, 'rb') as filehandle:
    dpo_ds = pickle.load(filehandle)

ckpt_filepath = do_train(
    dpo_ds=dpo_ds,
    tokenizer=tokenizer,
    model_name_or_path=pretrained_model_name,
    batch_size=4,
    gradient_accumulation_steps=8,
    max_epochs=3,
    lr=3e-5,
    deterministic=False
)
print(f"ckpt_filepath: {ckpt_filepath}")
