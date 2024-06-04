from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dotenv import load_dotenv
from typing import List
import lightning as pl
import requests
import warnings
import pickle
import torch
import wandb
import nltk
import os

from self_learning_open_generation import self_questioning_loop_open_generation_batched
from self_learning_utils import build_dataset


load_dotenv()
wandb.login()
nltk.download('punkt')

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
pl.seed_everything(47, workers=True)

torch.set_float32_matmul_precision("high")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pretrained_model_name = "mistralai/Mistral-7B-v0.1"
num_curator_workers = 1
verbose = False
do_information_retrieval = False
dir_result_dump = "result_dump"
dir_ds_dump = "dataset_dump"
os.makedirs(dir_result_dump, exist_ok=True)
os.makedirs(dir_ds_dump, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    use_fast=True,
    token=os.getenv('hf_personal_access_token')
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

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    token=os.getenv('hf_personal_access_token')
)
for param in model.parameters():
    param.requires_grad = False
model.to(torch.device("cuda"))

def prompt_fn(prompt):
    text = "<s>[INST] You are a student who is eager to learn about new things. [/INST]"
    text = text + "I am a student who is eager to learn about new things. I am aware of my lack of knowledge about some things.</s> "
    return text + f"[INST] {prompt} [/INST]"

def extract_response_fn(response):
    return response.split(" [/INST]")[-1]

def search_engine_fn(query: str) -> List[str]:
    if "GOOGLE_CUSTOM_SEARCH_URL" not in os.environ:
        raise ValueError("The environment variable GOOGLE_CUSTOM_SEARCH_URL is not set!")
    try:
        url = str(os.environ.get("GOOGLE_CUSTOM_SEARCH_URL")) + query
        response = requests.get(url)
        result = response.json()
        return [x["link"] for x in result["items"]]
    except Exception as e:
        warnings.warn("Error when searching on Google. | Error: " + str(e))
        pass
    return []


num_iteration = 3000

wandb_logger = wandb.init(
    project="SelfLearningFramework_v2",
    config={
        "batched_inference": True,
        "method": "open_generation",
        "pretrained_model_name": pretrained_model_name,
        "num_iteration": num_iteration
    }
)

batch_size = 256
outputs = self_questioning_loop_open_generation_batched(
    tokenizer, model, prompt_fn, extract_response_fn, batch_size,
    num_iteration=num_iteration, verbose=verbose, self_check_gpt_mode='NLI',
    use_cache=True, pretrained_model_name=pretrained_model_name
)

wandb.log({"curiosity_score": outputs["curiosity_score"], "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"], "self_learning_capability_score": outputs["self_learning_capability_score"]})
wandb.finish()

res_dump_filepath = dir_result_dump + "/res_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OpenGen.pickle"
with open(res_dump_filepath, 'wb') as dump_handle:
    pickle.dump(outputs, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)

if do_information_retrieval:
    questions_for_ds = [p["prompt"] for p in outputs["prompts_with_hallucination"]]
    ds = build_dataset(questions_for_ds, search_engine_fn, num_curator_workers)

    ds_dump_filepath = dir_ds_dump + "/ds_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OpenGen.pickle"
    with open(ds_dump_filepath, 'wb') as dump_handle:
        pickle.dump(ds, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
