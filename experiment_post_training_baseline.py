from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from dotenv import load_dotenv
import lightning as pl
from tqdm import tqdm
import statistics
import pickle
import torch
import wandb
import os

from self_learning_utils import HallucinationScorer, produce_passage, produce_samples, perplexity_evaluation, qa_evaluation_learned_data, qa_evaluation_benchmark


load_dotenv()
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
saved_model_path = "trained_1717082681_2775753"

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

with open(ds_filepath, 'rb') as dump_handle:
    ds = pickle.load(dump_handle)

with open(result_filepath, 'rb') as dump_handle:
    result = pickle.load(dump_handle)
questions_with_hallucination = result["prompts_with_hallucination"]

base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    token=os.getenv('hf_personal_access_token')
).to(torch.device('cuda'))
base_model.config.use_cache = False

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
print(f"tokenizer.eos_token: {tokenizer.eos_token}")


# BEFORE SELF-LEARNING EVAL
wandb_logger = wandb.init(
    project='self-learning-llm-benchmarking',
    name=f'BEFORE__{pretrained_model_name}'
)
before_training_avg_hallu = statistics.fmean(
    [x['average_score'] for x in questions_with_hallucination]
)
wandb_logger.log({'avg_hallucination': before_training_avg_hallu})
perplexity_evaluation(
    wandb_logger, base_model, tokenizer, batch_size=4
)
qa_evaluation_learned_data(
    wandb_logger, base_model, tokenizer, ds['prompt'], ds['chosen'], prompt_fn, extract_response_fn, batch_size=4
)
qa_evaluation_benchmark(
    wandb_logger, base_model, tokenizer, prompt_fn, extract_response_fn, batch_size=4
)
wandb_logger.finish()


peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
adapter_name = "injected_adapter"

trained_unmerged = PeftModel.from_pretrained(
    model=base_model,
    model_id = saved_model_path + "/injected_adapter/",
    adapter_name=adapter_name,
    is_trainable=False,
    config=peft_config,
    device_map="auto"
)
trained_model = trained_unmerged.merge_and_unload()

topics = [x["topics"] for x in questions_with_hallucination]
questions = [x["prompt"] for x in questions_with_hallucination]

tmp_passages = []
tmp_samples = []
print("Producing passages and samples....")
for idx in tqdm(range(len(questions))):
    passage = produce_passage(
        tokenizer, trained_model, prompt_fn, extract_response_fn,
        questions[idx], pretrained_model_name
    )
    print(passage, '\n')
    samples = produce_samples(
        tokenizer, trained_model, prompt_fn, extract_response_fn,
        questions[idx], pretrained_model_name
    )
    print(f"len(samples): {len(samples)}", '\n')
    tmp_passages.append(passage)
    tmp_samples.append(samples)
with open('storage/zzz_baseline_tmp_passages.pickle', 'wb') as dump_handle:
    pickle.dump(tmp_passages, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('storage/zzz_baseline_tmp_samples.pickle', 'wb') as dump_handle:
    pickle.dump(tmp_samples, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)

h_scorer = HallucinationScorer()
after_training_result = []
print("Performing hallucination scoring....")
for idx in tqdm(range(len(questions))):
    h_scorer_output = h_scorer.get_hallucination_score(
        topics[idx], questions[idx], tmp_passages[idx], tmp_samples[idx]
    )
    after_training_result.append(h_scorer_output)
    print(f"idx: {idx} | h_score: {h_scorer_output.average_score}")
    print()

before_training_avg_hallu = statistics.fmean(
    [x['average_score'] for x in questions_with_hallucination]
)
after_training_avg_hallu = statistics.fmean(
    [x.average_score for x in after_training_result]
)
print(f"before_training_avg_hallu: {before_training_avg_hallu}")
print(f"after_training_avg_hallu: {after_training_avg_hallu}")

after_training_result = [r.to_dict() for r in after_training_result]

with open("storage/baseline_after_training_result.pickle", "wb") as dump_handle:
    pickle.dump(after_training_result, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)


# AFTER SELF-LEARNING EVAL
wandb_logger = wandb.init(
    project='self-learning-llm-benchmarking',
    name=f'AFTER__{pretrained_model_name}'
)
wandb_logger.log({'avg_hallucination': after_training_avg_hallu})
perplexity_evaluation(
    wandb_logger, trained_model, tokenizer, batch_size=4
)
qa_evaluation_learned_data(
    wandb_logger, trained_model, tokenizer, ds['prompt'], ds['chosen'], prompt_fn, extract_response_fn, batch_size=4
)
qa_evaluation_benchmark(
    wandb_logger, trained_model, tokenizer, prompt_fn, extract_response_fn, batch_size=4
)
wandb_logger.finish()
