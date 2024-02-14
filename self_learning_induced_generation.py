from typing import Dict
from tqdm import tqdm

from self_learning_utils import (
    HallucinationScorer,
    generate_response,
    produce_passage,
    produce_samples,
    curiosity_measure,
    knowledge_limit_awareness_measure,
    self_learning_capability_measure
)


def self_questioning_loop_induced_generation(
        pretrained_model_name, tokenizer, model, prompt_fn, extract_response_fn,
        num_iteration=20, verbose=False
    ) -> Dict:
    h_scorer = HallucinationScorer()

    prompts_with_hallucination = []
    prompts_with_no_hallucination = []

    for iteration_idx in tqdm(range(num_iteration)):
        prompt1 = "Propose three topics that you would like to learn more about. Answer with only the three proposed topics concisely without elaboration."
        topics_to_learn = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompt1, pretrained_model_name)

        for question_word in ["WHAT", "WHO", "WHY", "WHERE", "WHEN", "HOW"]:
            prompt2 = f"Consider these topics: {topics_to_learn}. By using the word '{question_word}', propose only one question to query information about which you lack knowledge. Answer with only the proposed question concisely without elaboration."
            question_to_learn = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompt2, pretrained_model_name)

            passage = produce_passage(tokenizer, model, prompt_fn, extract_response_fn, question_to_learn, pretrained_model_name)
            samples = produce_samples(tokenizer, model, prompt_fn, extract_response_fn, question_to_learn, pretrained_model_name)

            h_scorer_output = h_scorer.get_hallucination_score(
                topics_to_learn, question_to_learn, passage, samples
            )

            if verbose:
                print("--------------------------------------------------")
                print(f"iteration {iteration_idx}")
                print("# hallucination score:\n", h_scorer_output.average_score)
                print("# topics:\n", topics_to_learn)
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

    self_learning_capability_score = self_learning_capability_measure(
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
        "self_learning_capability_score": self_learning_capability_score,
        "proposed_questions": proposed_questions,
        "proposed_questions_labels": proposed_questions_labels,
        "prompts_with_hallucination": prompts_with_hallucination,
        "prompts_with_no_hallucination": prompts_with_no_hallucination
    }
