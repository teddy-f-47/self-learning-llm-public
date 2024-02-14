from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm
import torch


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


def self_questioning_loop_embedding_shift(
        pretrained_model_name, tokenizer, model, prompt_fn, extract_response_fn,
        num_iteration=20, verbose=False
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
        prompt1 = "Propose three topics that you would like to learn more about. Answer with only the three proposed topics concisely without elaboration."
        topics_to_learn = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompt1, pretrained_model_name)

        initial_topic_embedding = embedder.encode(topics_to_learn, device="cpu", convert_to_numpy=False)
        randomized_shift = torch.randn(
            initial_topic_embedding.size(), requires_grad=False,
            dtype=initial_topic_embedding.dtype, device=initial_topic_embedding.device
        )

        initial_topic_embedding = F.normalize(initial_topic_embedding, p=1, dim=-1)
        randomized_shift = F.normalize(randomized_shift, p=1, dim=-1)
        shifted_topic_embedding = initial_topic_embedding + randomized_shift
        new_topics = random_transform_point_in_topic_embedding_space(topic_embedding_space=topic_embedding_space, all_nouns=all_nouns, initial_point=shifted_topic_embedding)

        prompt2 = f"Consider these topics: {new_topics}. Propose only one question to query information about which you lack knowledge. Answer with only the proposed question concisely without elaboration."
        question_to_learn = generate_response(tokenizer, model, prompt_fn, extract_response_fn, prompt2, pretrained_model_name)

        passage = produce_passage(tokenizer, model, prompt_fn, extract_response_fn, question_to_learn, pretrained_model_name)
        samples = produce_samples(tokenizer, model, prompt_fn, extract_response_fn, question_to_learn, pretrained_model_name)

        h_scorer_output = h_scorer.get_hallucination_score(
            new_topics, question_to_learn, passage, samples
        )

        if verbose:
            print("--------------------------------------------------")
            print(f"iteration {iteration_idx}")
            print("# hallucination score:\n", h_scorer_output.average_score)
            print("# initial topics:\n", topics_to_learn)
            print("# new topics:\n", new_topics)
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
