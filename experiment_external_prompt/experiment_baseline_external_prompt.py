from transformers import GenerationConfig
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

from self_learning_extrinsic import self_questioning_loop_extrinsic_inspiration
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

pretrained_model_name = "baseline"
num_curator_workers = 1
verbose = False
do_information_retrieval = False
dir_result_dump = "result_dump"
dir_ds_dump = "dataset_dump"
os.makedirs(dir_result_dump, exist_ok=True)
os.makedirs(dir_ds_dump, exist_ok=True)

tokenizer = lambda x: x
model = "baseline"

def prompt_fn(prompt):
    return prompt

def extract_response_fn(response):
    return response

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


num_iteration = 10

wandb_logger = wandb.init(
    project="SelfLearningFramework_v2",
    config={
        "batched_inference": True,
        "method": "external_prompt",
        "pretrained_model_name": pretrained_model_name,
        "num_iteration": num_iteration
    }
)

outputs = self_questioning_loop_extrinsic_inspiration(
    tokenizer, model, prompt_fn, extract_response_fn, num_iteration=num_iteration,
    use_cache=True, verbose=verbose, generation_config=GenerationConfig()
)

wandb.log({"curiosity_score": outputs["curiosity_score"], "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"], "self_learning_capability_score": outputs["self_learning_capability_score"]})
wandb.finish()

res_dump_filepath = dir_result_dump + "/res_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_ExtPrompt.pickle"
with open(res_dump_filepath, 'wb') as dump_handle:
    pickle.dump(outputs, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)

if do_information_retrieval:
    questions_for_ds = [p["prompt"] for p in outputs["prompts_with_hallucination"]]
    ds = build_dataset(questions_for_ds, search_engine_fn, num_curator_workers)

    ds_dump_filepath = dir_ds_dump + "/ds_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_ExtPrompt.pickle"
    with open(ds_dump_filepath, 'wb') as dump_handle:
        pickle.dump(ds, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
 