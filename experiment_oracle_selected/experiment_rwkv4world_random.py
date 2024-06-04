from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
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

from self_learning_oracle_selected import self_questioning_loop_oracle_selected_batched
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

pretrained_model_name = "RWKV/rwkv-4-world-7b"
num_curator_workers = 1
verbose = False
do_information_retrieval = False
dir_result_dump = "result_dump"
dir_ds_dump = "dataset_dump"
os.makedirs(dir_result_dump, exist_ok=True)
os.makedirs(dir_ds_dump, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    trust_remote_code=True,
    use_fast=True
)
model_config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
model_vocab_size = model_config.vocab_size
print(f"model_config.vocab_size: {model_config.vocab_size}")
print(f"default len(tokenizer): {len(tokenizer)}")

# if model_vocab_size > len(tokenizer):
    # model_vocab_size_diff = model_vocab_size - len(tokenizer)
    # additional_special_tokens = [f"<pad{token_id}>" for token_id in range(model_vocab_size_diff)]
    # tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    # print(f"extended len(tokenizer): {len(tokenizer)}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
print(f"tokenizer.eos_token: {tokenizer.eos_token}")

generation_config = GenerationConfig.from_model_config(model_config)
generation_config.eos_token_id = 0
generation_config.pad_token_id = 0
generation_config.top_p = 1.0
generation_config.top_k = 50

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.to(torch.device("cuda"))

def prompt_fn(input):
    instruction = "You are a student who is eager to learn about new things."
    # instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    return f"""Instruction: {instruction}

Input: {input}

Response:"""

def extract_response_fn(response):
    return response.split("Response:")[-1]

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
        "method": "oracle_selected",
        "pretrained_model_name": pretrained_model_name,
        "num_iteration": num_iteration
    }
)

batch_size = 256
outputs = self_questioning_loop_oracle_selected_batched(
    tokenizer, model, prompt_fn, extract_response_fn, batch_size,
    num_iteration=num_iteration, verbose=verbose, self_check_gpt_mode='NLI',
    use_cache=True,
    generation_config=generation_config
)

wandb.log({"curiosity_score": outputs["curiosity_score"], "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"], "self_learning_capability_score": outputs["self_learning_capability_score"]})
wandb.finish()

res_dump_filepath = dir_result_dump + "/res_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
with open(res_dump_filepath, 'wb') as dump_handle:
    pickle.dump(outputs, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)

if do_information_retrieval:
    questions_for_ds = [p["prompt"] for p in outputs["prompts_with_hallucination"]]
    ds = build_dataset(questions_for_ds, search_engine_fn, num_curator_workers)

    ds_dump_filepath = dir_ds_dump + "/ds_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
    with open(ds_dump_filepath, 'wb') as dump_handle:
        pickle.dump(ds, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
