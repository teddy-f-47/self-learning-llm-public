from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from peft import LoraConfig, PeftModel
from dotenv import load_dotenv
import lightning as pl
from tqdm import tqdm
import statistics
import pickle
import torch
import wandb
import copy
import os

from self_learning_utils import HallucinationScorer, produce_passage, produce_samples, perplexity_evaluation, qa_evaluation_learned_data, qa_evaluation_benchmark
from self_learning_training import get_router


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


"""
# BEFORE SELF-LEARNING EVAL
wandb_logger = wandb.init(
    project='self-learning-llm-benchmarking',
    name=f'BEFORE__ADAPTER__{pretrained_model_name}'
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
"""


peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
adapter_name = "injected_adapter"

unhallucinated_prompts = [prompt_fn(x['prompt']) for x in result['prompts_with_no_hallucination']]
unhallucinated_router_labels = [0] * len(unhallucinated_prompts)
hallucinated_prompts = [prompt_fn(x['prompt']) for x in result['prompts_with_hallucination']]
hallucinated_router_labels = [1] * len(hallucinated_prompts)

adapter_router = get_router(
    texts = unhallucinated_prompts + hallucinated_prompts,
    labels = unhallucinated_router_labels + hallucinated_router_labels,
    model = base_model,
    tokenizer = tokenizer,
    load_from_ckpt = None # 'storage/router_ckpt/1717144398_7720935/epoch=52-step=9964.ckpt'
)

class ModelWrapper():
    def __init__(self, base_model):
        self.base_model = base_model
        self.adapted_model = copy.deepcopy(base_model)
        self.adapter_dict = None
        self.adapter_router = None

    def eval(self):
        self.adapted_model.eval()

    def train(self):
        self.adapted_model.train()

    def set_adapter_router(self, adapter_dict, adapter_router):
        self.adapter_dict = adapter_dict
        self.adapter_router = adapter_router

    def disable_adapters(self):
        self.adapted_model = None
        self.adapted_model = copy.deepcopy(self.base_model)

    def enable_adapter(self, adapter_dir):
        base = copy.deepcopy(self.base_model)
        self.adapted_model = PeftModel.from_pretrained(
            model=base,
            model_id = adapter_dir,
            adapter_name=adapter_name,
            is_trainable=False,
            config=peft_config,
            device_map='auto'
        )
        self.adapted_model = self.adapted_model.merge_and_unload()

    def forward(self, *args, **kwargs):
        self.disable_adapters()
        input_ids = kwargs.get('input_ids') if 'input_ids' in kwargs else args[0]
        attention_mask = kwargs.get('attention_mask') if 'attention_mask' in kwargs else args[1]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[-1]
        with torch.no_grad():
            if seq_len < 256:
                pad_size = 256 - seq_len
                i = torch.nn.functional.pad(
                    input_ids, (0, pad_size), value=0
                )
                a = torch.nn.functional.pad(
                    attention_mask, (0, pad_size), value=0
                )
            else:
                i = input_ids[:, :256]
                a = attention_mask[:, :256]
            emb = self.adapted_model.forward(
                input_ids=i,
                attention_mask=a,
                output_hidden_states=True
            ).hidden_states
            emb = torch.stack(list(emb), dim=emb[0].dim())
            emb = emb.mean(dim=-1).float()
            if batch_size > 1:
                output = []
                for batch_idx in range(batch_size):
                    e = emb[batch_idx]
                    ro = self.adapter_router(e)
                    ro = int(torch.argmax(ro).item())
                    if ro == 0:
                        o = self.adapted_model.forward(
                            input_ids=input_ids[batch_idx].unsqueeze(0),
                            attention_mask=attention_mask[batch_idx].unsqueeze(0)
                        ).logits
                    else:
                        adapter_dir = self.adapter_dict[ro]
                        self.enable_adapter(adapter_dir)
                        o = self.adapted_model.forward(
                            input_ids=input_ids[batch_idx].unsqueeze(0),
                            attention_mask=attention_mask[batch_idx].unsqueeze(0)
                        ).logits
                    output.append(o)
                logits = torch.cat(output, dim=0)
                return CausalLMOutput(logits=logits)
            else:
                router_output = self.adapter_router(emb)
                router_output = int(torch.argmax(router_output).item())
                if router_output == 0:
                    return self.adapted_model.forward(*args, **kwargs)
                else:
                    adapter_dir = self.adapter_dict[router_output]
                    self.enable_adapter(adapter_dir)
                    return self.adapted_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.disable_adapters()
        input_ids = kwargs.get('input_ids') if 'input_ids' in kwargs else args[0]
        attention_mask = kwargs.get('attention_mask') if 'attention_mask' in kwargs else args[1]
        pad_token_id = kwargs.get('pad_token_id') if 'pad_token_id' in kwargs else 0
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[-1]
        with torch.no_grad():
            if seq_len < 256:
                pad_size = 256 - seq_len
                i = torch.nn.functional.pad(
                    input_ids, (0, pad_size), value=0
                )
                a = torch.nn.functional.pad(
                    attention_mask, (0, pad_size), value=0
                )
            else:
                i = input_ids[:, :256]
                a = attention_mask[:, :256]
            emb = self.adapted_model.forward(
                input_ids=i,
                attention_mask=a,
                output_hidden_states=True
            ).hidden_states
            emb = torch.stack(list(emb), dim=emb[0].dim())
            emb = emb.mean(dim=-1).float()
            if batch_size > 1:
                output = []
                for batch_idx in range(batch_size):
                    e = emb[batch_idx].unsqueeze(0)
                    ro = self.adapter_router(e)
                    ro = int(torch.argmax(ro).item())
                    print(f"ro: {ro}")
                    other_kwargs = {}
                    for k, v in kwargs.items():
                        if k != 'input_ids' and k != 'attention_mask':
                            other_kwargs[k] = v
                    i_len = input_ids[batch_idx].shape[-1]
                    if i_len < seq_len:
                        pad_size = seq_len - i_len
                        i = torch.nn.functional.pad(
                            input_ids[batch_idx], (0, pad_size), value=0
                        )
                        a = torch.nn.functional.pad(
                            attention_mask[batch_idx], (0, pad_size), value=0
                        )
                    else:
                        i = input_ids[batch_idx]
                        a = attention_mask[batch_idx]
                    if ro == 0:
                        o = self.adapted_model.generate(
                            input_ids=i.unsqueeze(0),
                            attention_mask=a.unsqueeze(0),
                            **other_kwargs
                        )
                    else:
                        adapter_dir = self.adapter_dict[ro]
                        self.enable_adapter(adapter_dir)
                        o = self.adapted_model.generate(
                            input_ids=i.unsqueeze(0),
                            attention_mask=a.unsqueeze(0),
                            **other_kwargs
                        )
                    output.append(o)
                output_lens = [o.shape[-1] for o in output]
                for o_idx in range(len(output)):
                    output_pad_size = max(output_lens) - output_lens[o_idx]
                    output[o_idx] = torch.nn.functional.pad(
                        output[o_idx], (0, output_pad_size), value=pad_token_id
                    )
                return torch.cat(output, dim=0)
            else:
                router_output = self.adapter_router(emb)
                router_output = int(torch.argmax(router_output).item())
                print(f"ro: {router_output}")
                if router_output == 0:
                    return self.adapted_model.generate(*args, **kwargs)
                else:
                    adapter_dir = self.adapter_dict[router_output]
                    self.enable_adapter(adapter_dir)
                    return self.adapted_model.generate(*args, **kwargs)

adapter_dict = {
    0: 'base_model',
    1: saved_model_path + "/injected_adapter/"
}
trained_model = ModelWrapper(base_model=base_model)
trained_model.set_adapter_router(adapter_dict, adapter_router)

topics = [x['topics'] for x in questions_with_hallucination]
questions = [x['prompt'] for x in questions_with_hallucination]

tmp_passages_dump_filepath = None
tmp_samples_dump_filepath = None
if tmp_passages_dump_filepath is None or tmp_samples_dump_filepath is None:
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
    with open('storage/zzz_tmp_passages.pickle', 'wb') as dump_handle:
        pickle.dump(tmp_passages, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('storage/zzz_tmp_samples.pickle', 'wb') as dump_handle:
        pickle.dump(tmp_samples, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(tmp_passages_dump_filepath, 'rb') as dumphandle:
        tmp_passages = pickle.load(dumphandle)
    with open(tmp_samples_dump_filepath, 'rb') as dumphandle:
        tmp_samples = pickle.load(dumphandle)

h_scorer = HallucinationScorer(device_str='cuda')
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

with open("storage/after_training_result.pickle", "wb") as dump_handle:
    pickle.dump(after_training_result, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)


# AFTER SELF-LEARNING EVAL
wandb_logger = wandb.init(
    project='self-learning-llm-benchmarking',
    name=f'AFTER__ADAPTER__{pretrained_model_name}'
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
