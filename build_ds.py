from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dotenv import load_dotenv
from typing import List
import lightning as pl
from math import ceil
import pandas as pd
import requests
import warnings
import pickle
import torch
import wandb
import nltk
import os

from self_learning_utils import build_dataset, build_dataset_chatgpt, score_article_relevance_with_chatgpt, check_accuracy_with_chatgpt, create_dpo_dataset


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

pretrained_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
num_curator_workers = 1
verbose = False
do_information_retrieval = False
do_relevance_scoring = False
do_ask_stronger_llm = False
do_create_dpo_dataset = True
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


with open('result_dump/res_mistralai_Mistral-7B-Instruct-v0_2_OracleSelected.pickle', 'rb') as dump_handle:
    outputs = pickle.load(dump_handle)

if do_information_retrieval:
    questions_for_ds = [p["prompt"] for p in outputs["prompts_with_hallucination"]]

    all_ds_filepaths = []
    batch_size = ceil(len(questions_for_ds) / 10)
    for batch_idx in range(10):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        ds = build_dataset(questions_for_ds[start_idx:end_idx], search_engine_fn, num_curator_workers, device='cpu')

        ds_dump_filepath = dir_ds_dump + "/dsGoogle" + str(batch_idx) + "_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
        all_ds_filepaths.append(ds_dump_filepath)
        with open(ds_dump_filepath, 'wb') as dump_handle:
            pickle.dump(ds, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_ds = []
    for filepath in all_ds_filepaths:
        with open(filepath, 'rb') as filehandle:
            ds = pickle.load(filehandle)
        all_ds.extend(ds)

    ds_dump_filepath = dir_ds_dump + "/dsGoogle_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
    with open(ds_dump_filepath, 'wb') as filehandle:
        pickle.dump(all_ds, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

if do_relevance_scoring:
    train_df_pickle_filepath = dir_ds_dump + "/trainGoogle_df_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"

    prompts = [x['prompt'] for x in outputs["prompts_with_hallucination"]]
    passages = [x['passage'] for x in outputs["prompts_with_hallucination"]]
    preds_df = pd.DataFrame({
        'prompt': prompts,
        'passage': passages
    }).sort_values(['prompt'], ascending=False).fillna('-')
    preds_df['prompt'] = preds_df['prompt'].apply(str).str.strip()
    preds_df['passage'] = preds_df['passage'].apply(str).str.strip()

    ds_dump_filepath = dir_ds_dump + "/dsGoogle_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
    with open(ds_dump_filepath, 'rb') as filehandle:
        ds = pickle.load(filehandle)
    docs_df = pd.DataFrame(ds)[['prompt', 'answer_score', 'answer_text', 'summary_text', 'full_text']]
    docs_df['prompt'] = docs_df['prompt'].apply(str).str.strip()
    docs_df['answer_text'] = docs_df['answer_text'].apply(str).str.strip()
    docs_df['summary_text'] = docs_df['summary_text'].apply(str).str.strip()
    docs_df['full_text'] = docs_df['full_text'].apply(str).str.strip()
    docs_df['answer_score'] = docs_df['answer_score'].astype(float)
    docs_df['LLM_judge_relevance'] = score_article_relevance_with_chatgpt(
        prompts=docs_df['prompt'].values.tolist(),
        texts=docs_df['full_text'].values.tolist()
    )

    train_df = pd.merge(docs_df, preds_df, on='prompt', how='left')
    prompts = train_df['prompt'].values.tolist()
    preds = train_df['passage'].values.tolist()
    refs = train_df['answer_text'].values.tolist()
    tmp_acc_scores, acc_scores = check_accuracy_with_chatgpt(
        questions=prompts, references=refs, predictions=preds
    )
    train_df['similarity_text'] = tmp_acc_scores
    train_df['similarity'] = acc_scores
    train_df.to_pickle(train_df_pickle_filepath)

if do_ask_stronger_llm:
    questions_for_ds = [p["prompt"] for p in outputs["prompts_with_hallucination"]]
    ds = build_dataset_chatgpt(questions_for_ds)

    ds_dump_filepath = dir_ds_dump + "/dsChatGPT_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
    with open(ds_dump_filepath, 'wb') as filehandle:
        pickle.dump(ds, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

if do_create_dpo_dataset:
    train_df_pickle_filepath = dir_ds_dump + "/trainChatGPT_df_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"

    if not os.path.exists(train_df_pickle_filepath):
        prompts = [x["prompt"] for x in outputs["prompts_with_hallucination"]]
        passages = [x["passage"] for x in outputs["prompts_with_hallucination"]]
        preds_df = pd.DataFrame({
            "prompt": prompts,
            "passage": passages
        }).sort_values(["prompt"], ascending=False).fillna('-')
        preds_df["prompt"] = preds_df["prompt"].apply(str).str.strip()
        preds_df["passage"] = preds_df["passage"].apply(str).str.strip()

        ds_dump_filepath = dir_ds_dump + "/dsChatGPT_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
        with open(ds_dump_filepath, 'rb') as filehandle:
            ds = pickle.load(filehandle)
        docs_df = pd.DataFrame(ds)
        docs_df["prompt"] = docs_df["prompt"].apply(str).str.strip()
        docs_df["answer_text"] = docs_df["answer_text"].apply(str).str.strip()

        train_df = pd.merge(docs_df, preds_df, on="prompt", how="left")
        prompts = train_df['prompt'].values.tolist()
        preds = train_df['passage'].values.tolist()
        refs = train_df['answer_text'].values.tolist()
        tmp_acc_scores, acc_scores = check_accuracy_with_chatgpt(
            questions=prompts, references=refs, predictions=preds
        )
        train_df['similarity_text'] = tmp_acc_scores
        train_df['similarity'] = acc_scores
        train_df.to_pickle(train_df_pickle_filepath)
    else:
        with open(train_df_pickle_filepath, 'rb') as filehandle:
            train_df = pickle.load(filehandle)
        prompts = train_df['prompt'].values.tolist()
        preds = train_df['passage'].values.tolist()
        refs = train_df['answer_text'].values.tolist()
        tmp_acc_scores, acc_scores = check_accuracy_with_chatgpt(
            questions=prompts, references=refs, predictions=preds
        )
        train_df['similarity_text'] = tmp_acc_scores
        train_df['similarity'] = acc_scores
        train_df.to_pickle(train_df_pickle_filepath)

    # yes: 32
    # no: 623
    # partly: 251
    # other: 16

    train_dpo_ds = create_dpo_dataset(train_df=train_df, prompt_format_fn=prompt_fn)
    train_dpo_ds_pickle_filepath = dir_ds_dump + "/trainChatGPT_ds_" + pretrained_model_name.replace("/", "_").replace(".", "_") + "_OracleSelected.pickle"
    with open(train_dpo_ds_pickle_filepath, 'wb') as filehandle:
        pickle.dump(train_dpo_ds, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
