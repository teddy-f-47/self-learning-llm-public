# Into the Unknown: Self-Learning Large Language Models

***Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko***

<https://arxiv.org/abs/2402.09147>

## Create a virtual environment (optional)

    - pip3 install virtualenv
    - python3 -m venv env

## Activate the virtual environment (optional)

    - For Windows: .\env\Scripts\activate
    - For Linux: source env/bin/activate

## Deactivate the virtual environment (optional)

    - deactivate

## Setup environment

`sudo apt-get update; sudo apt-get -y install python-dev libxml2-dev libxslt-dev; curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3; pip uninstall -y lxml; CFLAGS="-O0" pip install lxml; pip install -r requirements.txt; python -m spacy download en_core_web_sm; pip3 install --upgrade torch torchvision torchaudio; pip3 install datasets==2.16; pip3 install bitsandbytes loralib peft trl`

## Running an experiment

    - python3 -m DIRNAME.FILENAME
    - python3 -m experiment_open_generation.experiment_IntelNeuralChat_open
    - python3 -m experiment_induced_generation.experiment_IntelNeuralChat_induced
    - python3 -m experiment_oracle_selected.experiment_IntelNeuralChat_oracle
    - python3 -m experiment_external_prompt.experiment_IntelNeuralChat_external_prompt

## Results files

The `results` directory contains several subdirectories, which correspond to the experiments with different methods:

- `open_generation`: Self-Questioning with open question generation
- `induced_generation`: Self-Questioning with induced question generation
- `oracle_selected`: Self-Questioning with oracle-selected topic
- `external_prompt`: Self-Questioning with external topic

Each file in each of these subdirectories is a pickle file, which when loaded will give you a dictionary of the experiment output. The structure of the dictionary is as follows:

```
- "pretrained_model_name": str -- The name of the used model on HuggingFace
- "curiosity_score": float -- The calculated Curiosity Score
- "knowledge_limit_awareness_score": float -- The calculated Knowledge-Limit Awareness Score
- "self_learning_capability_score": float -- The calculated SLC Score
- "proposed_questions": List[str] -- List of all questions the model proposed; the length is equal to len(prompts_with_hallucination) plus len(prompts_with_no_hallucination)
- "proposed_questions_labels": List[int] -- List of question labels corresponding to proposed_questions, generated from hdbscan clustering during Knowledge-Limit Awareness Score calculation
- "prompts_with_hallucination": List[Dict] -- List of questions with hallucination, i.e. Q_H in the paper
    > "topics": str -- The topics, either proposed by the model or given from an external source, in a self-questioning iteration
    > "topics_embedding": numpy.ndarray -- The embedding of the string containing the proposed topics
    > "prompt": str -- The question proposed by the model in a self-questioning iteration
    > "prompt_embedding": numpy.ndarray -- The embedding of the proposed question
    > "passage": str -- The main passage produced by the model for hallucination scoring
    > "passage_sentences": List[str] -- The main passage but split into sentences, i.e. a list of sentences, which is actually used in the hallucination scoring
    > "samples": List[str] -- The samples produced by the model for hallucination scoring
    > "sentence_scores": numpy.ndarray -- The output from the hallucination scorer, which is an array of sentence-level hallucination scores
    > "average_score": float -- The average of sentence_scores, so the passage-level hallucination score
- "prompts_with_no_hallucination": List[Dict] -- List of questions with no hallucination, i.e. Q_NH in the paper
    > "topics": str -- The topics, either proposed by the model or given from an external source, in a self-questioning iteration
    > "topics_embedding": numpy.ndarray -- The embedding of the string containing the proposed topics
    > "prompt": str -- The question proposed by the model in a self-questioning iteration
    > "prompt_embedding": numpy.ndarray -- The embedding of the proposed question
    > "passage": str -- The main passage produced by the model for hallucination scoring
    > "passage_sentences": List[str] -- The main passage but split into sentences, i.e. a list of sentences, which is actually used in the hallucination scoring
    > "samples": List[str] -- The samples produced by the model for hallucination scoring
    > "sentence_scores": numpy.ndarray -- The output from the hallucination scorer, which is an array of sentence-level hallucination scores
    > "average_score": float -- The average of sentence_scores, so the passage-level hallucination score
```

## Running your own Self-Learning LLM

Refer to the experiment files and `experiment_training.py` to see how you can run Self-Questioning, Knowledge Searching, and Model Training. Refer to `experiment_post_training.py` if you would like to perform hallucination scoring on the model after training.

Some quick examples are provided below. Note that you can write your own `do_train` function if you want to customize the model training.

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import warnings


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name
)

def prompt_fn(prompt):
    return f"### System:\nYou are a student who is eager to learn about new things.\n### User:\n{prompt}\n### Assistant:\n"

def extract_response_fn(response):
    return response.split("### Assistant:\n")[-1]

def search_engine_fn(query: str) -> List[str]:
    if "CUSTOM_SEARCH_URL" not in os.environ:
        raise ValueError("The environment variable CUSTOM_SEARCH_URL is not set!")
    try:
        url = str(os.environ.get("CUSTOM_SEARCH_URL")) + query
        response = requests.get(url)
        result = response.json()
        return [x["link"] for x in result["items"]]
    except Exception as e:
        warnings.warn("Error in knowledge searching. | Error: " + str(e))
        pass
    return []
```

### Self-Learning LLM with Open Generation

```
from peft import LoraConfig, PeftModel
from self_learning_open_generation import self_questioning_loop_open_generation
from self_learning_utils import build_dataset
from self_learning_training import do_train

while True:
    outputs = self_questioning_loop_open_generation(
        pretrained_model_name, tokenizer, model, prompt_fn, extract_response_fn,
        num_iteration=100
    )
    print({
        "curiosity_score": outputs["curiosity_score"],
        "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"],
        "self_learning_capability_score": outputs["self_learning_capability_score"],
        "count_Q_H": len(outputs["prompts_with_hallucination"]),
        "count_Q_NH": len(outputs["prompts_with_no_hallucination"])
    })

    questions_for_ds = [p["prompt"] for p in outputs["prompts_with_hallucination"]]
    ds = build_dataset(questions_for_ds, search_engine_fn)

    ckpt_dir = do_train(
        ds=ds,
        questions_with_hallucination=questions_for_ds,
        prompt_format_fn=prompt_fn,
        extract_response_fn=extract_response_fn,
        tokenizer=tokenizer,
        model_name_or_path=pretrained_model_name,
        batch_size=4,
        max_epochs=10,
        lr=3e-5,
        deterministic=True
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapter_name = "injected_adapter"
    trained_unmerged = PeftModel.from_pretrained(
        model=model,
        model_id = ckpt_dir + "/injected_adapter",
        adapter_name=adapter_name,
        is_trainable=False,
        config=peft_config,
        device_map='auto'
    )
    model = trained_unmerged.merge_and_unload()
```

### Self-Learning LLM with Induced Generation

```
from peft import LoraConfig, PeftModel
from self_learning_induced_generation import self_questioning_loop_induced_generation
from self_learning_utils import build_dataset
from self_learning_training import do_train

while True:
    outputs = self_questioning_loop_induced_generation(
        pretrained_model_name, tokenizer, model, prompt_fn, extract_response_fn,
        num_iteration=100
    )
    print({
        "curiosity_score": outputs["curiosity_score"],
        "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"],
        "self_learning_capability_score": outputs["self_learning_capability_score"],
        "count_Q_H": len(outputs["prompts_with_hallucination"]),
        "count_Q_NH": len(outputs["prompts_with_no_hallucination"])
    })
    ...
```

### Self-Learning LLM with Oracle-Selected

```
from peft import LoraConfig, PeftModel
from self_learning_oracle_selected import self_questioning_loop_oracle_selected
from self_learning_utils import build_dataset
from self_learning_training import do_train

while True:
    outputs = self_questioning_loop_oracle_selected(
        pretrained_model_name, tokenizer, model, prompt_fn, extract_response_fn,
        num_iteration=100
    )
    print({
        "curiosity_score": outputs["curiosity_score"],
        "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"],
        "self_learning_capability_score": outputs["self_learning_capability_score"],
        "count_Q_H": len(outputs["prompts_with_hallucination"]),
        "count_Q_NH": len(outputs["prompts_with_no_hallucination"])
    })
    ...
```

### Self-Learning LLM with External Prompt

```
from peft import LoraConfig, PeftModel
import os
from self_learning_extrinsic import self_questioning_loop_extrinsic_inspiration
from self_learning_utils import build_dataset
from self_learning_training import do_train

os.environ["SERP_API_KEY"] = "..."

while True:
    outputs = self_questioning_loop_extrinsic_inspiration(
        pretrained_model_name, tokenizer, model, prompt_fn, extract_response_fn,
        num_iteration=100
    )
    print({
        "curiosity_score": outputs["curiosity_score"],
        "knowledge_limit_awareness_score": outputs["knowledge_limit_awareness_score"],
        "self_learning_capability_score": outputs["self_learning_capability_score"],
        "count_Q_H": len(outputs["prompts_with_hallucination"]),
        "count_Q_NH": len(outputs["prompts_with_no_hallucination"])
    })
    ...
```

## Disclaimer

This repository was created for reproducibility purposes of our paper. All work is intended only for scientific research. We are not responsible for the actions of other parties who use this repository.

## Citation

```
@misc{ferdinan2024unknown,
    title={Into the Unknown: Self-Learning Large Language Models}, 
    author={Teddy Ferdinan and Jan Kocoń and Przemysław Kazienko},
    year={2024},
    eprint={2402.09147},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
