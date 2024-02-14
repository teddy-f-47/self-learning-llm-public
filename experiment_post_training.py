from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
import lightning as pl
from tqdm import tqdm
import statistics
import pickle
import torch
import wandb
import os

from self_learning_utils import HallucinationScorer, produce_passage, produce_samples


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
result_filepath = "results/open_generation/res_Intel_neural-chat-7b-v3-3_OpenGen.pickle"
saved_model_path = "trained_1707757461_244057"

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

with open(result_filepath, 'rb') as dump_handle:
    result = pickle.load(dump_handle)
questions_with_hallucination = result["prompts_with_hallucination"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
base_model.config.use_cache = False
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
adapter_name = "injected_adapter"

topics = [x['topics'] for x in questions_with_hallucination]
questions = [x['prompt'] for x in questions_with_hallucination]
trained_unmerged = PeftModel.from_pretrained(
    model=base_model,
    model_id = saved_model_path + "/injected_adapter",
    adapter_name=adapter_name,
    is_trainable=False,
    config=peft_config,
    device_map='auto'
)
trained_model = trained_unmerged.merge_and_unload()

tmp_passages = []
tmp_samples = []
print("Producing passages and samples....")
for idx in tqdm(range(len(questions))):
    passage = produce_passage(
        tokenizer, trained_model, prompt_fn, extract_response_fn,
        questions[idx], pretrained_model_name
    )
    samples = produce_samples(
        tokenizer, trained_model, prompt_fn, extract_response_fn,
        questions[idx], pretrained_model_name
    )
    tmp_passages.append(passage)
    tmp_samples.append(samples)

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

with open("after_training_result.pickle", "wb") as dump_handle:
    pickle.dump(after_training_result, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
