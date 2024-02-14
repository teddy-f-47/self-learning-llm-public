# Into the Unknown: Self-Learning Large Language Models

***Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko***

*Paper to be submitted to arXiv and ACL 2024*

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
