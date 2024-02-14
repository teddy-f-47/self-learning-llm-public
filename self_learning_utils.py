from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig, pipeline
from typing import List, Dict, Any, Union, Callable
from nltk.corpus import wordnet as wn
from scipy.spatial import KDTree
import torch.nn.functional as F
from newspaper import Article
from threading import Thread
from pathlib import Path
from tqdm import tqdm
import numpy as np
import statistics
import warnings
import hdbscan
import pickle
import random
import spacy
import torch
import copy
import nltk
import os


class HallucinationScorerResult():
    def __init__(
            self, topics: str, prompt: str, passage: str, passage_sentences: List[str],
            samples: List[str], sentence_scores: np.ndarray, average_score: np.ndarray
        ) -> None:
        self.topics = topics
        self.prompt = prompt
        self.passage = passage
        self.passage_sentences = passage_sentences
        self.samples = samples
        self.sentence_scores = sentence_scores
        self.average_score = average_score

    def to_dict(self) -> Dict:
        return {
            "topics": self.topics,
            "prompt": self.prompt,
            "passage": self.passage,
            "passage_sentences": self.passage_sentences,
            "samples": self.samples,
            "sentence_scores": self.sentence_scores,
            "average_score": self.average_score
        }

class HallucinationScorer():
    def __init__(
            self, spacy_model_name: str = "en_core_web_sm",
            mode: str = "NLI", device_str: str = "cuda"
        ) -> None:
        self.processor = spacy.load(spacy_model_name)
        if device_str == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "Device was set to 'cuda', but cuda is not available.\
                HallucinationScorer is using 'cpu' instead."
            )
            device_str = "cpu"
        if mode != "NLI":
            warnings.warn(
                "The selected mode is not implemented yet.\
                HallucinationScorer is using 'NLI' instead."
            )
        self.checker = SelfCheckNLI(device=torch.device(device_str))

    def get_hallucination_score(
            self, topics: str, prompt: str, passage: str, samples: List[str]
        ) -> HallucinationScorerResult:
        passage_sentences = [st.text.strip() for st in self.processor(passage).sents]
        sentence_scores = self.checker.predict(
            sentences=passage_sentences, sampled_passages=samples
        )
        if np.isnan(sentence_scores).any():
            warnings.warn("Warning: Found NaN sent_score. Using numpy.nanmean().")
            average_score = np.nanmean(sentence_scores)
        else:
            average_score = sentence_scores.mean()
        return HallucinationScorerResult(
            topics=topics,
            prompt=prompt,
            passage=passage,
            passage_sentences=passage_sentences,
            samples=samples,
            sentence_scores=sentence_scores,
            average_score=average_score
        )

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None) -> None:
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self) -> None:
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args) -> Any:
        Thread.join(self, *args)
        return self._return

class CuratedItem():
    def __init__(
            self, prompt: str, answer_text: str, summary_text: str,
            full_text: str, answer_score: float, source_url: str
        ) -> None:
        self.prompt = prompt
        self.answer_text = answer_text
        self.summary_text = summary_text
        self.full_text = full_text
        self.answer_score = answer_score
        self.source_url = source_url

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "answer_text": self.answer_text,
            "summary_text": self.summary_text,
            "full_text": self.full_text,
            "answer_score": self.answer_score,
            "source_url": self.source_url
        }

class Curator():
    def __init__(
            self,
            search_engine_fn: Callable,
            summarization_min_length: int = 1,
            summarization_max_length: int = 256,
            num_workers: int = 2
        ) -> None:
        self.question_answerer = pipeline(
            "question-answering", tokenizer="deepset/tinyroberta-squad2",
            model="deepset/tinyroberta-squad2", framework="pt"
        )
        self.summarizer = pipeline(
            "summarization", tokenizer="facebook/bart-large-cnn",
            model="facebook/bart-large-cnn", framework="pt"
        )
        self.search_engine_fn = search_engine_fn
        self.summarization_min_length = summarization_min_length
        self.summarization_max_length = summarization_max_length
        self.num_workers = num_workers

    def read_webpage(self, prompt: str, link: str) -> Union[CuratedItem, None]:
        try:
            article = Article(link)
            article.download()
            article.parse()
            article.nlp()
            full_text = "-" if not article.text else article.text
            qa_output = self.question_answerer(
                {"question": prompt, "context": full_text}
            )
            summary_output = self.summarizer(
                full_text,
                min_length=self.summarization_min_length,
                max_length=self.summarization_max_length,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            answer_text = qa_output["answer"]
            answer_score = qa_output["score"]
            summary_text = summary_output[0]["summary_text"]
            return CuratedItem(
                prompt=prompt, answer_text=answer_text, summary_text=summary_text,
                full_text=full_text, answer_score=answer_score, source_url=link
            )
        except Exception as e:
            warnings.warn("Error when reading a webpage. | Error: " + str(e))
            pass
        return None

    def construct_dataset(self, prompts_with_hallucination: List[str]) -> List[CuratedItem]:
        output = []
        for prompt in tqdm(prompts_with_hallucination):
            links = self.search_engine_fn(prompt)
            if self.num_workers > 1:
                start = 0
                end = self.num_workers
                num_links_read = 0
                while num_links_read < len(links):
                    links_batch = links[start:end]
                    workers = []
                    for link in links_batch:
                        workers.append(
                            ThreadWithReturnValue(target=self.read_webpage, args=(prompt, link))
                        )
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        curated_item = worker.join()
                        if curated_item is not None:
                            output.append(curated_item)
                    start = end
                    end = end + self.num_workers
                    num_links_read = num_links_read + len(links_batch)
            else:
                for link in links:
                    curated_item = self.read_webpage(prompt, link)
                    if curated_item is not None:
                        output.append(curated_item)
        return output

def generate(tokenizer, model, prompt_fn, extract_response_fn, prompt, generation_config):
    prompt = prompt_fn(prompt)
    input_tokens = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(torch.device("cuda"))
    with torch.inference_mode():
        generated_ids = model.generate(**input_tokens, generation_config=generation_config)
    raw_pred = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    pred = extract_response_fn(raw_pred[0])
    return pred

def generate_response(
        tokenizer, model, prompt_fn, extract_response_fn, prompt,
        pretrained_model_name=None, generation_config=None
    ):
    if (
        (pretrained_model_name is None and generation_config is None) or
        (pretrained_model_name is not None and generation_config is not None)
    ):
        raise ValueError("You must provide either pretrained_model_name or generation_config.")
    if generation_config is None:
        question_generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name, max_new_tokens=64, do_sample=True, temperature=1.0
        )
    else:
        question_generation_config = copy.deepcopy(generation_config)
        question_generation_config.max_new_tokens = 64
        question_generation_config.do_sample = True
        question_generation_config.temperature = 1.0
    return generate(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, question_generation_config
    )

def generate_response_with_temperature(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, temperature,
        pretrained_model_name=None, generation_config=None
    ):
    if (
        (pretrained_model_name is None and generation_config is None) or
        (pretrained_model_name is not None and generation_config is not None)
    ):
        raise ValueError("You must provide either pretrained_model_name or generation_config.")
    if generation_config is None:
        question_generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name, max_new_tokens=64, do_sample=True, temperature=temperature
        )
    else:
        question_generation_config = copy.deepcopy(generation_config)
        question_generation_config.max_new_tokens = 64
        question_generation_config.do_sample = True
        question_generation_config.temperature = temperature
    return generate(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, question_generation_config
    )

def produce_passage(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, pretrained_model_name=None
    ):
    model.eval()
    passage_generation_config = GenerationConfig.from_pretrained(
        pretrained_model_name, do_sample=False, repetition_penalty=1.1,
        min_new_tokens=1, max_new_tokens=128
    )
    return generate(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, passage_generation_config
    )

def produce_samples(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, pretrained_model_name=None
    ):
    model.eval()
    samples_generation_config = GenerationConfig.from_pretrained(
        pretrained_model_name, do_sample=True, repetition_penalty=1.1,
        temperature=1.0, num_return_sequences=10, min_new_tokens=1, max_new_tokens=128
    )
    return generate(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, samples_generation_config
    )

def get_nouns_and_embeddings(embedder: SentenceTransformer) -> Dict:
    nltk.download('omw-1.4')
    dir_storage = "storage"
    os.makedirs(dir_storage, exist_ok=True)
    pickl_all_nouns_filepath = dir_storage + "/E_all_nouns.pickle"
    pickl_all_embeddings_filepath = dir_storage + "/E_all_embeddings.pickle"
    if Path(pickl_all_nouns_filepath).is_file() and Path(pickl_all_embeddings_filepath).is_file():
        with open(pickl_all_nouns_filepath, "rb") as dump_handle:
            all_nouns = pickle.load(dump_handle)
        with open(pickl_all_embeddings_filepath, "rb") as dump_handle:
            all_embeddings = pickle.load(dump_handle)
    else:
        all_nouns = []
        for synset in wn.all_synsets("n"):
            lemma_names = [str(lemma.name()) for lemma in synset.lemmas()]
            lemma_descs = [str(synset.definition()) for lemma in synset.lemmas()]
            lemms = [n + " ### " + d for n, d in zip(lemma_names, lemma_descs)]
            all_nouns.extend(lemms)
        all_nouns = list(set(all_nouns))
        all_embeddings = embedder.encode(all_nouns, device="cpu", convert_to_numpy=True)
        with open(pickl_all_nouns_filepath, "wb") as dump_handle:
            pickle.dump(all_nouns, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pickl_all_embeddings_filepath, "wb") as dump_handle:
            pickle.dump(all_embeddings, dump_handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {'all_nouns': all_nouns, 'all_embeddings': all_embeddings}

def get_topic_embedding_space(all_embeddings: List) -> KDTree:
    all_embeddings = torch.tensor(all_embeddings)
    all_embeddings = F.normalize(all_embeddings, p=1, dim=-1)
    return KDTree(all_embeddings)

def random_transform_point_in_topic_embedding_space(
        topic_embedding_space: KDTree, all_nouns: List, initial_point: torch.Tensor
    ) -> str:
    emb_distances, emb_indices = topic_embedding_space.query(initial_point.cpu(), k=10)
    new_topic_candidates = [all_nouns[idx].split(" ### ")[0] for idx in emb_indices]
    new_topic = random.sample(new_topic_candidates, k=3)
    new_topic = [x.replace("_", " ") for x in new_topic]
    new_topic = ", ".join(new_topic)
    return new_topic

def calculate_brevity_coefficient(text_lengths: List[float]) -> float:
    assumed_ideal_sentence_len = 100
    avg_text_len_diff = statistics.fmean([abs(x - assumed_ideal_sentence_len) for x in text_lengths])
    if avg_text_len_diff <= 50:
        return 1
    avg_text_len_diff = 1e-7 if avg_text_len_diff == 0 else avg_text_len_diff
    brevity_coefficient = 1 - (avg_text_len_diff / assumed_ideal_sentence_len) + 0.5
    return brevity_coefficient

def curiosity_measure(
        proposed_questions: List[str], num_question_generation: int
    ) -> float:
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    embeddings = embedder.encode(proposed_questions, device="cpu", convert_to_numpy=True)
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(embeddings)
    num_clusters = clusterer.labels_.max()
    num_outliers = len([x for x in clusterer.labels_ if x == -1])
    num_unique_questions = num_clusters + num_outliers
    text_lens = [len(x) for x in proposed_questions]
    brevity_coefficient = calculate_brevity_coefficient(text_lens)
    print("# curiosity_measure")
    print(f"num_clusters: {num_clusters}")
    print(f"num_outliers: {num_outliers}")
    print(f"num_unique_questions: {num_unique_questions}")
    print(f"num_question_generation: {num_question_generation}")
    return brevity_coefficient * num_unique_questions / num_question_generation, clusterer.labels_

def knowledge_limit_awareness_measure(num_prompts_with_hallucination: int, num_total_prompts: int) -> float:
    print("# knowledge_limit_awareness_measure")
    print(f"num_prompts_with_hallucination: {num_prompts_with_hallucination}")
    print(f"num_total_prompts: {num_total_prompts}")
    print()
    return num_prompts_with_hallucination / num_total_prompts

def self_learning_capability_measure(curiosity_score: float, knowledge_limit_awareness_score: float) -> float:
    return statistics.fmean([curiosity_score, knowledge_limit_awareness_score])

def build_dataset(
        prompts_with_hallucination: List[str], search_engine_fn: Callable, num_workers: int = 2
    ) -> List[Dict]:
    print("Building dataset....")
    curator = Curator(search_engine_fn=search_engine_fn, num_workers=num_workers)
    output = curator.construct_dataset(prompts_with_hallucination)
    return [o.to_dict() for o in output]
