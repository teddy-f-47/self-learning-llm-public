from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset as HFDataset
import bitsandbytes as bnb
from trl import DPOTrainer
from tqdm import tqdm
import pandas as pd
import statistics
import pickle
import torch
import wandb
import time

from self_learning_utils import HallucinationScorer, produce_passage, produce_samples


def create_train_ds(ds, questions_with_hallucination, prompt_format_fn):
    prompts = [x['prompt'] for x in questions_with_hallucination]
    passages = [x['passage'] for x in questions_with_hallucination]
    preds_df = pd.DataFrame({
        'prompt': prompts,
        'passage': passages
    }).sort_values(['prompt'], ascending=False).fillna('-')
    preds_df['prompt'] = preds_df['prompt'].apply(str).str.strip()
    preds_df['passage'] = preds_df['passage'].apply(str).str.strip()
    docs_df = pd.DataFrame(ds)[['prompt', 'answer_score', 'answer_text', 'summary_text']]
    docs_df['prompt'] = docs_df['prompt'].apply(str).str.strip()
    docs_df['answer_text'] = docs_df['answer_text'].apply(str).str.strip()
    docs_df['summary_text'] = docs_df['summary_text'].apply(str).str.strip()
    docs_df['answer_score'] = docs_df['answer_score'].astype(float)
    all_df = pd.merge(docs_df, preds_df, on='prompt', how='left')
    all_df = all_df.sort_values(['prompt', 'answer_score'], ascending=False)
    all_df = all_df.groupby('prompt').head(1)

    print(all_df.head(20))

    prompt_list = [prompt_format_fn(x) for x in all_df['prompt'].values.tolist()]
    tmp_answer = all_df['answer_text'].values.tolist()
    tmp_summary = all_df['summary_text'].values.tolist()
    chosen_list = [a.strip() + '. ' + s.strip() for a, s in zip(tmp_answer, tmp_summary)]
    rejected_list = all_df['passage'].values.tolist()
    return HFDataset.from_dict({
        'prompt': prompt_list,
        'chosen': chosen_list,
        'rejected': rejected_list
    })

def do_train(
        ds,
        questions_with_hallucination,
        prompt_format_fn,
        extract_response_fn,
        tokenizer,
        model_name_or_path,
        batch_size = 1,
        max_epochs = 80,
        lr = 3e-5,
        deterministic = True
    ):
    train_ds = create_train_ds(ds, questions_with_hallucination, prompt_format_fn)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
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
    model = get_peft_model(base_model, peft_config, adapter_name)
    model_ref = get_peft_model(base_model, peft_config, adapter_name)

    training_args = TrainingArguments(
        output_dir="tmp_model",
        overwrite_output_dir=True,
        full_determinism=deterministic,
        do_train=True,
        do_eval=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        optim='adamw_bnb_8bit',
        learning_rate=lr,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bf16 = True,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.1,
        num_train_epochs=max_epochs,
        save_total_limit=1,
        report_to='wandb',
        disable_tqdm=False,
        push_to_hub=False
    )
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=128
    )
    dpo_trainer.train()

    saved_model_path = "trained_" + str(time.time()).replace('.', '_')
    dpo_trainer.save_model(saved_model_path)
    wandb.finish()

    return saved_model_path
