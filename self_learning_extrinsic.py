from serpapi import GoogleSearch as GoogleSerpAPI
from dotenv import load_dotenv
from typing import Dict
from tqdm import tqdm
import pickle
import time
import os

from self_learning_utils import (
    HallucinationScorer,
    generate_response,
    produce_passage,
    produce_samples,
    curiosity_measure,
    knowledge_limit_awareness_measure,
    self_learning_capability_measure
)


def prompt_engineering_for_question_generation(input: str) -> str:
    return f"Consider these topics: {input}. Propose only one question about something that you have little or no knowledge. Answer with only the proposed question concisely without elaboration."

def prompt_engineering_for_passage_or_samples(input: str) -> str:
    return f"Answer concisely with only one sentence. {input}"

def get_google_trends_trending_now(iteration_idx, use_cache=True) -> str:
    load_dotenv()
    os.makedirs("storage", exist_ok=True)
    cache_filepath = f"storage/cached_google_trends{iteration_idx}.pickle"
    if not use_cache or not os.path.isfile(cache_filepath):
        if "SERP_API_KEY" not in os.environ:
            raise ValueError("The environment variable SERP_API_KEY is not set!")
        serp_api_params = {
            "engine": "google_trends_trending_now",
            "frequency": "realtime",
            "geo": "US",
            "api_key": str(os.environ.get("SERP_API_KEY"))
        }
        serp_api_output = GoogleSerpAPI(serp_api_params)
        gtrends = serp_api_output.get_dict()
        realtime_searches = gtrends["realtime_searches"]
        topics = [rs["title"] for rs in realtime_searches]
        topics = topics[:100]
        with open(cache_filepath, 'wb') as dump_handle:
            pickle.dump(topics, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
        cache_is_used = False
    else:
        print(f'Using cache {cache_filepath}...')
        with open(cache_filepath, 'rb') as dump_handle:
            topics = pickle.load(dump_handle)
        cache_is_used = True
    return topics, cache_is_used

# it doesn't make sense to make the batched generation version for this function
# because we need to wait for at least one hour for each new set of trends anyway
def self_questioning_loop_extrinsic_inspiration(
        tokenizer, model, prompt_fn, extract_response_fn, num_iteration=2, use_cache=True,
        verbose=False, pretrained_model_name=None, generation_config=None
    ) -> Dict:
    h_scorer = HallucinationScorer()

    prompts_with_hallucination = []
    prompts_with_no_hallucination = []

    for iteration_idx in tqdm(range(num_iteration)):
        list_of_new_topics, cache_is_used = get_google_trends_trending_now(iteration_idx=iteration_idx, use_cache=use_cache)
        print(f"Number of received trending topics: {len(list_of_new_topics)}")
        subprocessing_start_time = time.time()

        for new_topic_idx, new_topics in enumerate(list_of_new_topics):
            prompt2 = prompt_engineering_for_question_generation(new_topics)
            question_to_learn = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompt2, pretrained_model_name, generation_config)

            passage = produce_passage(tokenizer, model, prompt_fn, extract_response_fn, prompt_engineering_for_passage_or_samples(question_to_learn), pretrained_model_name, generation_config)
            samples = produce_samples(tokenizer, model, prompt_fn, extract_response_fn, prompt_engineering_for_passage_or_samples(question_to_learn), pretrained_model_name, generation_config)

            h_scorer_output = h_scorer.get_hallucination_score(
                new_topics, question_to_learn, passage, samples
            )

            if verbose:
                print("--------------------------------------------------")
                print(f"iteration {iteration_idx}, topic {new_topic_idx}")
                print("# hallucination score:\n", h_scorer_output.average_score)
                print("# topics:\n", new_topics)
                print("# question:\n", question_to_learn)
                print("# main prediction:\n", passage)
                print()

            if h_scorer_output.average_score > 0.5:
                prompts_with_hallucination.append(h_scorer_output)
            else:
                prompts_with_no_hallucination.append(h_scorer_output)

        subprocessing_end_time = time.time()
        print(f'cache_is_used: {cache_is_used}')

        # Real-time Google Trends is updated hourly, so let's sleep 30 minutes
        # (assuming the processing of each batch of trends also takes about 30 minutes, 30+30=60)
        if iteration_idx == num_iteration-1:
            break # don't sleep if it is already last iteration
        if not cache_is_used:
            elapsed_time = subprocessing_end_time - subprocessing_start_time
            time_until_next_iter = 3600 - elapsed_time + 10
            print(f"Already elapsed time: {elapsed_time} seconds")
            print(f"Time until next iteration: {time_until_next_iter} seconds")
            if time_until_next_iter > 0:
                time.sleep(time_until_next_iter)
        else:
            print("Cache was used, hence going to the next iteration right away.")

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
