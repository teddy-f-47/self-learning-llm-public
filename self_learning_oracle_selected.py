from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from typing import Dict
from math import floor
from tqdm import tqdm
import pickle
import torch
import os

from self_learning_utils import (
    HallucinationScorer,
    generate_response,
    produce_passage,
    produce_samples,
    get_nouns_and_embeddings,
    get_topic_embedding_space,
    random_transform_point_in_topic_embedding_space,
    curiosity_measure,
    knowledge_limit_awareness_measure,
    self_learning_capability_measure
)


def prompt_engineering_for_question_generation(input: str) -> str:
    return f"Consider one of these topics: {input}. Propose only one question about something that you have little or no knowledge. Answer with only the proposed question concisely without elaboration."

def prompt_engineering_for_passage_or_samples(input: str) -> str:
    return f"Answer concisely with only one sentence. {input}"

def self_questioning_loop_oracle_selected(
        tokenizer, model, prompt_fn, extract_response_fn, num_iteration=20,
        verbose=False, pretrained_model_name=None, generation_config=None
    ) -> Dict:
    h_scorer = HallucinationScorer()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    nouns_and_embeddings = get_nouns_and_embeddings(embedder=embedder)
    all_nouns = nouns_and_embeddings["all_nouns"]
    all_embeddings = nouns_and_embeddings["all_embeddings"]
    topic_embedding_space = get_topic_embedding_space(all_embeddings=all_embeddings)

    prompts_with_hallucination = []
    prompts_with_no_hallucination = []

    for iteration_idx in tqdm(range(num_iteration)):
        random_point = torch.randn([384], requires_grad=False, device=torch.device('cpu'))
        random_point = F.normalize(random_point, p=1, dim=-1)
        new_topics = random_transform_point_in_topic_embedding_space(topic_embedding_space=topic_embedding_space, all_nouns=all_nouns, initial_point=random_point)

        prompt2 = prompt_engineering_for_question_generation(new_topics)
        question_to_learn = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompt2, pretrained_model_name, generation_config)

        passage = produce_passage(tokenizer, model, prompt_fn, extract_response_fn, prompt_engineering_for_passage_or_samples(question_to_learn), pretrained_model_name, generation_config)
        samples = produce_samples(tokenizer, model, prompt_fn, extract_response_fn, prompt_engineering_for_passage_or_samples(question_to_learn), pretrained_model_name, generation_config)

        h_scorer_output = h_scorer.get_hallucination_score(
            new_topics, question_to_learn, passage, samples
        )

        if verbose:
            print("--------------------------------------------------")
            print(f"iteration {iteration_idx}")
            print("# hallucination score:\n", h_scorer_output.average_score)
            print("# topics:\n", new_topics)
            print("# question:\n", question_to_learn)
            print("# main prediction:\n", passage)
            print()

        if h_scorer_output.average_score > 0.5:
            prompts_with_hallucination.append(h_scorer_output)
        else:
            prompts_with_no_hallucination.append(h_scorer_output)

    proposed_questions = []
    proposed_questions.extend([p.prompt for p in prompts_with_hallucination])
    proposed_questions.extend([p.prompt for p in prompts_with_no_hallucination])
    curiosity_score, proposed_questions_labels = curiosity_measure(
        proposed_questions, len(proposed_questions)
    )

    knowledge_limit_awareness_score = knowledge_limit_awareness_measure(
        len(prompts_with_hallucination),
        len(prompts_with_hallucination)+len(prompts_with_no_hallucination)
    )

    self_learning_capability_score, brevity_coefficient = self_learning_capability_measure(
        proposed_questions,
        curiosity_score,
        knowledge_limit_awareness_score
    )

    prompts_with_hallucination = [p.to_dict() for p in prompts_with_hallucination]
    prompts_with_no_hallucination = [p.to_dict() for p in prompts_with_no_hallucination]

    print(f"curiosity_score: {curiosity_score}")
    print(f"knowledge_limit_awareness_score: {knowledge_limit_awareness_score}")
    print(f"self_learning_capability_score: {self_learning_capability_score}")

    return {
        "pretrained_model_name": pretrained_model_name,
        "curiosity_score": curiosity_score,
        "knowledge_limit_awareness_score": knowledge_limit_awareness_score,
        "brevity_coefficient": brevity_coefficient,
        "self_learning_capability_score": self_learning_capability_score,
        "proposed_questions": proposed_questions,
        "proposed_questions_labels": proposed_questions_labels,
        "prompts_with_hallucination": prompts_with_hallucination,
        "prompts_with_no_hallucination": prompts_with_no_hallucination
    }

def self_questioning_loop_oracle_selected_batched(
        tokenizer, model, prompt_fn, extract_response_fn, batch_size, num_iteration=20,
        verbose=False, self_check_gpt_mode='NLI', use_cache=False,
        pretrained_model_name=None, generation_config=None
    ) -> Dict:
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    nouns_and_embeddings = get_nouns_and_embeddings(embedder=embedder)
    all_nouns = nouns_and_embeddings["all_nouns"]
    all_embeddings = nouns_and_embeddings["all_embeddings"]
    topic_embedding_space = get_topic_embedding_space(all_embeddings=all_embeddings)

    if not use_cache or not os.path.exists('storage/oracle_topics.pickle'):
        topics = []
        for iteration_idx in range(num_iteration):
            random_point = torch.randn([384], requires_grad=False, device=torch.device('cpu'))
            random_point = F.normalize(random_point, p=1, dim=-1)
            new_topics = random_transform_point_in_topic_embedding_space(topic_embedding_space=topic_embedding_space, all_nouns=all_nouns, initial_point=random_point)
            topics.append(new_topics)
        with open('storage/oracle_topics.pickle', 'wb') as dumpfile:
            pickle.dump(topics, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('storage/oracle_topics.pickle', 'rb') as dumpfile:
            topics = pickle.load(dumpfile)
    print(f"Total number of topics: {len(topics)}")

    if not use_cache or not os.path.exists('storage/oracle_questions.pickle'):
        prompts = [prompt_engineering_for_question_generation(t) for t in topics]
        questions = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompts, pretrained_model_name, generation_config, batch_size)
        with open('storage/oracle_questions.pickle', 'wb') as dumpfile:
            pickle.dump(questions, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('storage/oracle_questions.pickle', 'rb') as dumpfile:
            questions = pickle.load(dumpfile)
    print(f"Total number of questions: {len(questions)}")

    questions = [prompt_engineering_for_passage_or_samples(q) for q in questions]

    if not use_cache or not os.path.exists('storage/oracle_passage_pool.pickle'):
        passage_pool = produce_passage(tokenizer, model, prompt_fn, extract_response_fn, questions, pretrained_model_name, generation_config, batch_size)
        with open('storage/oracle_passage_pool.pickle', 'wb') as dumpfile:
            pickle.dump(passage_pool, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('storage/oracle_passage_pool.pickle', 'rb') as dumpfile:
            passage_pool = pickle.load(dumpfile)
    print(f"Total number of passages: {len(passage_pool)}")

    if not use_cache or not os.path.exists('storage/oracle_samples_pool.pickle'):
        samples_generation_batch_size = floor(batch_size/10)
        samples_generation_batch_size = samples_generation_batch_size if samples_generation_batch_size > 1 else 2
        samples_pool = produce_samples(tokenizer, model, prompt_fn, extract_response_fn, questions, pretrained_model_name, generation_config, samples_generation_batch_size)
        with open('storage/oracle_samples_pool.pickle', 'wb') as dumpfile:
            pickle.dump(samples_pool, dumpfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('storage/oracle_samples_pool.pickle', 'rb') as dumpfile:
            samples_pool = pickle.load(dumpfile)
    print(f"Total number of samples lists: {len(samples_pool)}")
    print(f"Total number of samples in each list: {len(samples_pool[0])}")

    h_scorer = HallucinationScorer(mode=self_check_gpt_mode)
    prompts_with_hallucination = []
    prompts_with_no_hallucination = []
    for current_topic_idx in tqdm(range(len(topics))):
        t = topics[current_topic_idx]
        q = questions[current_topic_idx]
        passage = passage_pool[current_topic_idx]
        samples = samples_pool[current_topic_idx]

        h_scorer_output = h_scorer.get_hallucination_score(t, q, passage, samples)

        if verbose:
            print(f"topic {current_topic_idx} | hallucination: {h_scorer_output.average_score}")

        if h_scorer_output.average_score > 0.5:
            prompts_with_hallucination.append(h_scorer_output)
        else:
            prompts_with_no_hallucination.append(h_scorer_output)

    proposed_questions = []
    proposed_questions.extend([p.prompt for p in prompts_with_hallucination])
    proposed_questions.extend([p.prompt for p in prompts_with_no_hallucination])
    curiosity_score, proposed_questions_labels = curiosity_measure(
        proposed_questions, len(proposed_questions)
    )

    knowledge_limit_awareness_score = knowledge_limit_awareness_measure(
        len(prompts_with_hallucination),
        len(prompts_with_hallucination)+len(prompts_with_no_hallucination)
    )

    self_learning_capability_score, brevity_coefficient = self_learning_capability_measure(
        proposed_questions,
        curiosity_score,
        knowledge_limit_awareness_score
    )

    prompts_with_hallucination = [p.to_dict() for p in prompts_with_hallucination]
    prompts_with_no_hallucination = [p.to_dict() for p in prompts_with_no_hallucination]

    print(f"curiosity_score: {curiosity_score}")
    print(f"knowledge_limit_awareness_score: {knowledge_limit_awareness_score}")
    print(f"self_learning_capability_score: {self_learning_capability_score}")

    return {
        "pretrained_model_name": pretrained_model_name,
        "curiosity_score": curiosity_score,
        "knowledge_limit_awareness_score": knowledge_limit_awareness_score,
        "brevity_coefficient": brevity_coefficient,
        "self_learning_capability_score": self_learning_capability_score,
        "proposed_questions": proposed_questions,
        "proposed_questions_labels": proposed_questions_labels,
        "prompts_with_hallucination": prompts_with_hallucination,
        "prompts_with_no_hallucination": prompts_with_no_hallucination
    }
