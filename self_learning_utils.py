from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckLLMPrompt
from torchmetrics.functional import pairwise_cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig, pipeline
from typing import List, Dict, Union, Callable
from datasets import Dataset as HFDataset
from nltk.corpus import wordnet as wn
from torch.nn import CrossEntropyLoss
from scipy.spatial import KDTree
import torch.nn.functional as F
from dotenv import load_dotenv
from newspaper import Article
from threading import Thread
from openai import OpenAI
from pathlib import Path
from math import ceil
from tqdm import tqdm
import numpy as np
import statistics
import datasets
import evaluate
import warnings
import hdbscan
import pickle
import random
import spacy
import torch
import copy
import math
import nltk
import time
import os


class PerplexityWithCustomModel(evaluate.Metric):
    def _info(self):
        metric_desc = """
This function is copied from the evaluate library version 0.4.1. The token-level perplexity
calculation is exactly the same as the original. The differences are only related to how
the model is loaded. In addition, additional metrics are provided: token-level negative
log-likelihood, word-level perplexity, and character-level perplexity.
"""
        inputs_desc = """
Args:
    model: model used for calculating Perplexity.
    tokenizer: tokenizer for the model.
    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available.
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
"""
        return evaluate.MetricInfo(
            module_type="metric",
            description=metric_desc,
            citation="-",
            inputs_description=inputs_desc,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=[
                "https://huggingface.co/docs/transformers/perplexity",
                "https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py",
                "https://sjmielke.com/comparing-perplexities.htm"
            ],
        )

    def _compute(
        self, predictions, model, tokenizer, batch_size: int = 16,
        add_start_token: bool = True, device=None, max_length=None
    ):
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model.eval()

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            if len(existing_special_tokens) > 0:
            # assign one of the special tokens to also be the pad token
                tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
            else:
                print("WARNING: No special token was found. Using token with ID 0 as padding token. Please make sure that this token is indeed viable for padding, otherwise you will get bad results.")
                tokenizer.add_special_tokens({"pad_token": tokenizer.decode(0)})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]
        txt_char_counts = torch.Tensor([len(x.strip()) for x in predictions]).to(device)
        txt_word_counts = torch.Tensor([len(x.split()) for x in predictions]).to(device)

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        dump_txt_char_lens = []
        dump_txt_word_lens = []
        nlls = []
        token_level_ppls = []
        char_level_ppls = []
        word_level_ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in evaluate.logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            txt_char_count = txt_char_counts[start_index:end_index]
            txt_word_count = txt_word_counts[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model.forward(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous().transpose(1, 2)
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            nll = (loss_fct(shift_logits, shift_labels) * shift_attention_mask_batch).sum(1)

            perplexity_batch = torch.exp(nll / shift_attention_mask_batch.sum(1))
            char_level_perplexity_batch = torch.exp(nll / txt_char_count)
            word_level_perplexity_batch = torch.exp(nll / txt_word_count)

            dump_txt_char_lens += txt_char_count.tolist()
            dump_txt_word_lens += txt_word_count.tolist()
            nlls += nll.tolist()
            token_level_ppls += perplexity_batch.tolist()
            char_level_ppls += char_level_perplexity_batch.tolist()
            word_level_ppls += word_level_perplexity_batch.tolist()

        return {
            "txt_char_lens": dump_txt_char_lens,
            "mean_txt_char_lens": np.mean(dump_txt_char_lens),
            "txt_word_lens": dump_txt_word_lens,
            "mean_txt_word_lens": np.mean(dump_txt_word_lens),
            "NLL": nlls,
            "mean_NLL": np.mean(nlls),
            "token_PPL": token_level_ppls,
            "mean_token_PPL": np.mean(token_level_ppls),
            "char_PPL": char_level_ppls,
            "mean_char_PPL": np.mean(char_level_ppls),
            "word_PPL": word_level_ppls,
            "mean_word_PPL": np.mean(word_level_ppls)
        }

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
            llm_checker_name: str = "openchat/openchat_3.5",
            mode: str = "NLI", device_str: str = "cuda"
        ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "Device was set to 'cuda', but cuda is not available.\
                HallucinationScorer is using 'cpu' instead."
            )
            device_str = "cpu"
        if mode == "NLI":
            self.checker = SelfCheckNLI(
                device=torch.device(device_str)
            )
        elif mode == "LLM":
            self.checker = SelfCheckLLMPrompt(
                llm_checker_name, device=torch.device(device_str)
            )
        else:
            warnings.warn(
                "The selected mode is not implemented yet.\
                HallucinationScorer is using 'NLI' instead."
            )
            self.checker = SelfCheckNLI(device=torch.device(device_str))
        self.processor = spacy.load(spacy_model_name)

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
            num_workers: int = 2,
            device: str = 'cpu'
        ) -> None:
        self.question_answerer = pipeline(
            "question-answering", tokenizer="deepset/tinyroberta-squad2",
            model="deepset/tinyroberta-squad2", framework="pt", device=device
        )
        self.summarizer = pipeline(
            "summarization", tokenizer="facebook/bart-large-cnn",
            model="facebook/bart-large-cnn", framework="pt", device=device
        )
        self.search_engine_fn = search_engine_fn
        self.summarization_min_length = summarization_min_length
        self.summarization_max_length = summarization_max_length
        self.num_workers = num_workers

    def read_webpage(
            self, prompt: str, link: str, result_pool=None, result_index=None
        ) -> Union[CuratedItem, None]:
        try:
            article = Article(link)
            article.download()
            article.parse()
            article.nlp()
            full_text = "-" if not article.text else article.text
            if full_text != "-":
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
                output = CuratedItem(
                    prompt=prompt, answer_text=answer_text, summary_text=summary_text,
                    full_text=full_text, answer_score=answer_score, source_url=link
                )
                if result_pool is not None and result_index is not None:
                    result_pool[result_index] = output
                return output
        except Exception as e:
            warnings.warn("Error when reading a webpage. | Error: " + str(e))
            pass
        if result_pool is not None and result_index is not None:
            result_pool[result_index] = None
        return None

    def construct_dataset(self, prompts_with_hallucination: List[str]) -> List[CuratedItem]:
        output = []
        for prompt in tqdm(prompts_with_hallucination):
            links = self.search_engine_fn(prompt)
            if self.num_workers > 1:
                num_links_read = 0
                while num_links_read < len(links):
                    start = num_links_read
                    end = start + self.num_workers
                    links_batch = links[start:end]
                    threads = [None] * len(links_batch)
                    results = [None] * len(links_batch)
                    for i in range(len(links_batch)):
                        threads[i] = Thread(
                            target=self.read_webpage,
                            args=(prompt, links_batch[i], results, i)
                        )
                        threads[i].start()
                    time.sleep(1)
                    for i in range(len(links_batch)):
                        threads[i].join()
                    for r in results:
                        if r is not None:
                            output.append(r)
                    num_links_read = num_links_read + len(links_batch)
            else:
                for link in links:
                    curated_item = self.read_webpage(prompt, link)
                    if curated_item is not None:
                        output.append(curated_item)
        return output

def generate(
        tokenizer, model, prompt_fn, extract_response_fn, prompt: str,
        generation_config
    ):
    if model == 'baseline':
        pred = []
        for pred_idx in range(generation_config.num_return_sequences):
            pred.append(prompt + '...')
    else:
        model.eval()
        prompt = prompt_fn(prompt)
        input_tokens = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(torch.device("cuda"))
        with torch.inference_mode():
            generated_ids = model.generate(**input_tokens, generation_config=generation_config)
        raw_pred = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        pred = [extract_response_fn(rp) for rp in raw_pred]
        all_special_tokens = []
        try:
            all_special_tokens.extend(tokenizer.all_special_tokens)
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer._additional_special_tokens.items()])
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer.added_tokens_decoder.items()])
        except:
            pass
        for spectok in all_special_tokens:
            pred = [p.replace(spectok, '') for p in pred]
    return pred

def batch_generate(
        tokenizer, model, prompt_fn, extract_response_fn, prompts: List[str],
        generation_config, batch_size=4
    ):
    if model == 'baseline':
        preds = []
        for prompt in prompts:
            for pred_idx in range(generation_config.num_return_sequences):
                preds.append(prompt + '...')
    else:
        model.eval()
        prompts = [prompt_fn(p) for p in prompts]
        batches = np.array_split(np.asarray(prompts), math.ceil(len(prompts)/batch_size))
        raw_preds = []
        for batch in tqdm(batches):
            input_tokens = tokenizer(
                batch.tolist(), padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(torch.device("cuda"))
            with torch.inference_mode():
                generated_ids = model.generate(
                    **input_tokens, generation_config=generation_config
                )
            output_seqs = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            raw_preds.extend(output_seqs)
        preds = [extract_response_fn(p) for p in raw_preds]
        all_special_tokens = []
        try:
            all_special_tokens.extend(tokenizer.all_special_tokens)
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer._additional_special_tokens.items()])
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer.added_tokens_decoder.items()])
        except:
            pass
        for spectok in all_special_tokens:
            preds = [p.replace(spectok, '') for p in preds]
    return preds

def generate_response(
        tokenizer, model, prompt_fn, extract_response_fn, prompt,
        pretrained_model_name=None, generation_config=None, batch_size=None
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
    if batch_size is None or type(prompt) == str:
        if type(prompt) == list:
            prompt = prompt[0]
        output = generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            question_generation_config
        )[0]
    else:
        output = batch_generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            question_generation_config, batch_size
        )
    return output

def generate_response_with_temperature(
        tokenizer, model, prompt_fn, extract_response_fn, prompt, temperature,
        pretrained_model_name=None, generation_config=None, batch_size=None
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
    if batch_size is None or type(prompt) == str:
        if type(prompt) == list:
            prompt = prompt[0]
        output = generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            question_generation_config
        )[0]
    else:
        output = batch_generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            question_generation_config, batch_size
        )
    return output

def produce_passage(
        tokenizer, model, prompt_fn, extract_response_fn, prompt,
        pretrained_model_name=None, generation_config=None, batch_size=None
    ):
    if (
        (pretrained_model_name is None and generation_config is None) or
        (pretrained_model_name is not None and generation_config is not None)
    ):
        raise ValueError("You must provide either pretrained_model_name or generation_config.")
    if generation_config is None:
        passage_generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name, do_sample=False, repetition_penalty=1.1,
            min_new_tokens=1, max_new_tokens=128
        )
    else:
        passage_generation_config = copy.deepcopy(generation_config)
        passage_generation_config.do_sample = False
        passage_generation_config.repetition_penalty = 1.1
        passage_generation_config.min_new_tokens = 1
        passage_generation_config.max_new_tokens = 128
    if batch_size is None or type(prompt) == str:
        if type(prompt) == list:
            prompt = prompt[0]
        output = generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            passage_generation_config
        )[0]
    else:
        output = batch_generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            passage_generation_config, batch_size
        )
    return output

def produce_samples(
        tokenizer, model, prompt_fn, extract_response_fn, prompt,
        pretrained_model_name=None, generation_config=None, batch_size=None
    ):
    if (
        (pretrained_model_name is None and generation_config is None) or
        (pretrained_model_name is not None and generation_config is not None)
    ):
        raise ValueError("You must provide either pretrained_model_name or generation_config.")
    if generation_config is None:
        samples_generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name, do_sample=True, repetition_penalty=1.1,
            temperature=1.0, num_return_sequences=10, min_new_tokens=1, max_new_tokens=128
        )
    else:
        samples_generation_config = copy.deepcopy(generation_config)
        samples_generation_config.do_sample = True
        samples_generation_config.repetition_penalty = 1.1
        samples_generation_config.temperature = 1.0
        samples_generation_config.num_return_sequences = 10
        samples_generation_config.min_new_tokens = 1
        samples_generation_config.max_new_tokens = 128
    if batch_size is None or type(prompt) == str:
        if type(prompt) == list:
            prompt = prompt[0]
        output = generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            samples_generation_config
        )
    else:
        output_pool = batch_generate(
            tokenizer, model, prompt_fn, extract_response_fn, prompt,
            samples_generation_config, batch_size
        )
        output = []
        for start_idx in range(0, len(output_pool), 10):
            end_idx = start_idx + 10
            samples = output_pool[start_idx:end_idx]
            output.append(samples)
    return output

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
    if avg_text_len_diff >= 100:
        return 0
    if avg_text_len_diff <= 50:
        return 1
    brevity_coefficient = 1 - (avg_text_len_diff / assumed_ideal_sentence_len) + 0.5
    return brevity_coefficient

def curiosity_measure(
        proposed_questions: List[str], num_question_generation: int
    ) -> float:
    literally_unique_questions = list(set(proposed_questions))
    literally_unique_count = len(literally_unique_questions)
    if literally_unique_count > 10: # ensure at least 10 questions are "literally unique" to do clustering
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
        embeddings = embedder.encode(proposed_questions, device="cpu", convert_to_numpy=True)
        clusterer = hdbscan.HDBSCAN(cluster_selection_method='leaf', allow_single_cluster=True)
        clusterer.fit(embeddings)
        num_clusters = clusterer.labels_.max() + 1 # labels start from 0
        num_outliers = len([x for x in clusterer.labels_ if x == -1])
        num_unique_questions = num_clusters + num_outliers
        cluster_labels = clusterer.labels_.tolist()
    else:
        num_unique_questions = literally_unique_count
        num_clusters = literally_unique_count
        num_outliers = 0
        cluster_dict = {q: idx for idx, q in enumerate(literally_unique_questions)}
        cluster_labels = [cluster_dict[q] for q in proposed_questions]
    print("# curiosity_measure")
    print(f"num_clusters: {num_clusters}")
    print(f"num_outliers: {num_outliers}")
    print(f"num_unique_questions: {num_unique_questions}")
    print(f"num_question_generation: {num_question_generation}")
    return num_unique_questions / num_question_generation, cluster_labels

def knowledge_limit_awareness_measure(num_prompts_with_hallucination: int, num_total_prompts: int) -> float:
    print("# knowledge_limit_awareness_measure")
    print(f"num_prompts_with_hallucination: {num_prompts_with_hallucination}")
    print(f"num_total_prompts: {num_total_prompts}")
    print()
    return num_prompts_with_hallucination / num_total_prompts

def self_learning_capability_measure(proposed_questions: List[str], curiosity_score: float, knowledge_limit_awareness_score: float) -> float:
    text_lens = [len(x) for x in proposed_questions]
    brevity_coefficient = calculate_brevity_coefficient(text_lens)
    return brevity_coefficient * statistics.fmean([curiosity_score, knowledge_limit_awareness_score]), brevity_coefficient

def score_article_relevance_with_chatgpt(prompts: List[str], texts: List[str]):
    assert len(prompts) == len(texts)
    load_dotenv()
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY_PERSONAL'))
    output = []
    print("-- score_article_relevance_with_chatgpt --")
    for idx in tqdm(range(len(prompts))):
        message = '###PROMPT: ' + prompts[idx] + '\n\n'
        message = message + '###CONTEXT: ' + texts[idx]
        resp = openai_client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            seed=42,
            temperature=0,
            n=1,
            messages=[
                {"role": "system", "content": "Act as an expert in evaluating the relevance of an article for answering a question. As an input, you will receive two texts:\n\n###PROMPT: the question to be answered;\n\n###CONTEXT: the article providing the context to answer the question.\n\nDo you accept ###CONTEXT as a relevant article to answer the question contained in ###PROMPT?\n\nYES - if the article ###CONTEXT can be considered relevant to ###PROMPT\n\nNO - if the article is completely irrelevant to ###PROMPT\n\nPARTLY - if the the article is partially relevant to ###PROMPT"},
                {"role": "user", "content": message}
            ]
        )
        raw_resp = resp.choices[0].message.content
        output.append(raw_resp)
    return output

def cosine_similarity_score(model, tokenizer, preds: List[str], refs: List[str], device):
    assert len(preds) == len(refs)
    output = []
    for idx in tqdm(range(len(preds))):
        pred = preds[idx]
        ref = refs[idx]
        x_tokens = tokenizer(
            [pred],
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            max_length=4096,
            return_tensors="pt",
            return_attention_mask=True
        ).to(device)
        y_tokens = tokenizer(
            [ref],
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            max_length=4096,
            return_tensors="pt",
            return_attention_mask=True
        ).to(device)

        model.eval()
        with torch.no_grad():
            x = model(**x_tokens, output_hidden_states=True).hidden_states[0]
            x = x.mean(dim=-1).float()
            y = model(**y_tokens, output_hidden_states=True).hidden_states[0]
            y = y.mean(dim=-1).float()
        score = pairwise_cosine_similarity(x, y, reduction='mean')
        output.append(score.item())
    return output

def build_dataset(
        prompts_with_hallucination: List[str], search_engine_fn: Callable, num_workers: int = 2, device: str = 'cpu'
    ) -> List[Dict]:
    print("Building dataset....")
    curator = Curator(search_engine_fn=search_engine_fn, num_workers=num_workers, device=device)
    output = curator.construct_dataset(prompts_with_hallucination)
    return [o.to_dict() for o in output]

def ask_answers_from_chatgpt(prompts: List[str]):
    load_dotenv()
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY_PERSONAL'))
    output = []
    print("-- ask_answers_from_chatgpt --")
    for idx in tqdm(range(len(prompts))):
        message = prompts[idx]
        resp = openai_client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            seed=42,
            temperature=0,
            n=1,
            messages=[
                {"role": "system", "content": "Act as an expert capable of answering questions from various domains."},
                {"role": "user", "content": message}
            ]
        )
        raw_resp = resp.choices[0].message.content
        output.append(raw_resp)
    return output

def check_accuracy_with_chatgpt(questions, references, predictions):
    load_dotenv()
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY_PERSONAL'))
    output = []
    print("-- check_accuracy_with_chatgpt --")
    for idx in tqdm(range(len(predictions))):
        message = '###PROMPT: ' + questions[idx] + '\n\n'
        message = message + '###GOLD_ANSWER: ' + references[idx] + '\n\n'
        message = message + '###SYSTEM_ANSWER: ' + predictions[idx]
        resp = openai_client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            seed=42,
            temperature=0,
            n=1,
            messages=[
                {"role": "system", "content": "Act as an expert in evaluating the correctness of other models' answers. As an input, you will receive three texts:\n\n##PROMPT: task definition for the model;\n\n##GOLD_ANSWER: the correct answer prepared by a human expert;\n\n##SYSTEM_ANSWER: the answer of the model you are evaluating.\n\nDo you accept ##SYSTEM_ANSWER as the correct answer that captures the meaning contained in ##GOLD_ANSWER?\n\nYES - if the answer ##SYSTEM_ANSWER can be considered correct relative to ##GOLD_ANSWER\n\nNO - if the answer ##SYSTEM_ANSWER is completely different from ##GOLD_ANSWER\n\nPARTLY - if the answer ##SYSTEM_ANSWER can be considered partially correct relative to ##GOLD_ANSWER"},
                {"role": "user", "content": message}
            ]
        )
        raw_resp = resp.choices[0].message.content
        output.append(raw_resp)
    acc_scores = []
    for idx in range(len(output)):
        txt = output[idx].lower().strip()
        if txt.startswith('yes'):
            acc_scores.append(1.0)
        elif txt.startswith('no'):
            acc_scores.append(0.0)
        elif txt.startswith('partly'):
            acc_scores.append(0.5)
        else:
            acc_scores.append(-99.0)
    return output, acc_scores

def build_dataset_chatgpt(prompts_with_hallucination: List[str]):
    prompts = list(set(prompts_with_hallucination))
    answers = ask_answers_from_chatgpt(prompts)
    output = []
    for idx in tqdm(range(len(prompts))):
        output.append({
            'prompt': prompts[idx],
            'answer_text': answers[idx]
        })
    return output

def create_dpo_dataset(train_df, prompt_format_fn):
    prompt_list = [prompt_format_fn(x) for x in train_df['prompt'].values.tolist()]
    similarity_score_list = train_df['similarity'].values.tolist()
    passage_list = train_df['passage'].values.tolist()
    answer_text_list = train_df['answer_text'].values.tolist()

    chosen_list = []
    rejected_list = []
    for idx in tqdm(range(len(prompt_list))):
        similarity = similarity_score_list[idx]
        if similarity > 0.5:
            chosen_list.append(passage_list[idx])
            rejected_list.append('-')
        else:
            chosen_list.append(answer_text_list[idx])
            rejected_list.append(passage_list[idx])

    chosen_list = [x.strip() for x in chosen_list]
    rejected_list = [x.strip() for x in rejected_list]

    return HFDataset.from_dict({
        'prompt': prompt_list,
        'chosen': chosen_list,
        'rejected': rejected_list
    })

def perplexity_evaluation(
        wandb_logger, model, tokenizer, batch_size=16, text_word_len=256
    ):
    def find_nth_word(haystack: str, needle: str = ' ', n: int = 13) -> int:
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start
    ds = datasets.load_dataset('teddy-f-47/wikipedia_simple_20231201', split='train')
    ds = ds.shuffle(seed=42)
    ds = ds.flatten_indices()
    ds = ds.select(list(range(1000)))
    ds = ds['text']
    ds = [x.strip() for x in ds if x is not None and len(x.strip()) > 1]
    ds = [x[:find_nth_word(x, ' ', text_word_len)] for x in ds]
    perplexity = PerplexityWithCustomModel()
    with torch.no_grad():
        results = perplexity.compute(
            model=model,
            tokenizer=tokenizer,
            add_start_token=False,
            predictions=ds,
            batch_size=batch_size
        )
    wandb_logger.log(results)

def qa_evaluation_learned_data(
        wandb_logger, model, tokenizer, questions_list, gold_answers_list, prompt_fn, extract_response_fn, batch_size=16
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    prompt_list = [prompt_fn(x) for x in questions_list]

    preds = []
    num_batches = ceil(len(prompt_list) / batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = prompt_list[start:end]
        tokens = tokenizer(
            batch,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)
        generated_ids = model.generate(
            **tokens, max_new_tokens=64, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        decoded = [extract_response_fn(x) for x in decoded]
        all_special_tokens = []
        try:
            all_special_tokens.extend(tokenizer.all_special_tokens)
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer._additional_special_tokens.items()])
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer.added_tokens_decoder.items()])
        except:
            pass
        for spectok in all_special_tokens:
            decoded = [p.replace(spectok, '') for p in decoded]
        preds.extend(decoded)

    bleu = evaluate.load('bleu')
    bleu_res = bleu.compute(predictions=preds, references=[[x] for x in gold_answers_list])
    wandb_logger.log({f'learned_{key}': val for key, val in bleu_res.items()})

    rouge = evaluate.load('rouge')
    rouge_res = rouge.compute(predictions=preds, references=gold_answers_list)
    wandb_logger.log({f'learned_{key}': val for key, val in rouge_res.items()})

    oracle_scores_texts, oracle_scores = check_accuracy_with_chatgpt(questions_list, gold_answers_list, preds)
    oracle_scores_texts = {k: v for k, v in enumerate(oracle_scores_texts)}
    oracle_scores = {k: v for k, v in enumerate(oracle_scores)}
    wandb_logger.log({
        'learned_oracle_acc_text': oracle_scores_texts,
        'learned_oracle_acc': oracle_scores
    })

def qa_evaluation_benchmark(
        wandb_logger, model, tokenizer, prompt_fn, extract_response_fn, batch_size=16
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    ds = datasets.load_dataset('yahma/alpaca-cleaned', split='train')
    ds = ds.shuffle(seed=42)
    ds = ds.flatten_indices()
    ds = ds.select(list(range(1000)))

    def prompt_no_input(row):
        content = row['instruction']
        return prompt_fn(content)

    def prompt_input(row):
        content = f"\n\n###Instruction: {row['instruction']}\n\n###Input: {row['input']}"
        return prompt_fn(content)

    def create_prompt(row):
        return prompt_no_input(row) if row['input'] == '' else prompt_input(row)

    ds = ds.map(lambda x: {'formatted_prompt': create_prompt(x)})
    formatted_prompts = ds['formatted_prompt']
    answers = ds['output']

    preds = []
    num_batches = ceil(len(formatted_prompts) / batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = formatted_prompts[start:end]
        tokens = tokenizer(
            batch,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)
        generated_ids = model.generate(
            **tokens, max_new_tokens=64, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        decoded = [extract_response_fn(x) for x in decoded]
        all_special_tokens = []
        try:
            all_special_tokens.extend(tokenizer.all_special_tokens)
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer._additional_special_tokens.items()])
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer.added_tokens_decoder.items()])
        except:
            pass
        for spectok in all_special_tokens:
            decoded = [p.replace(spectok, '') for p in decoded]
        preds.extend(decoded)

    bleu = evaluate.load('bleu')
    bleu_res = bleu.compute(predictions=preds, references=[[x] for x in answers])
    wandb_logger.log({f'benchmark_{key}': val for key, val in bleu_res.items()})

    rouge = evaluate.load('rouge')
    rouge_res = rouge.compute(predictions=preds, references=answers)
    wandb_logger.log({f'benchmark_{key}': val for key, val in rouge_res.items()})

    oracle_scores_texts, oracle_scores = check_accuracy_with_chatgpt(formatted_prompts, answers, preds)
    oracle_scores_texts = {k: v for k, v in enumerate(oracle_scores_texts)}
    oracle_scores = {k: v for k, v in enumerate(oracle_scores)}
    wandb_logger.log({
        'benchmark_oracle_acc_text': oracle_scores_texts,
        'benchmark_oracle_acc': oracle_scores
    })

def evaluate_hallucination(
        topics, questions, tokenizer, model, prompt_fn, extract_response_fn,
        pretrained_model_name=None, generation_config=None
    ):
    tmp_passages = []
    tmp_samples = []
    print("Producing passages and samples....")
    for idx in tqdm(range(len(questions))):
        passage = produce_passage(
            tokenizer, model, prompt_fn, extract_response_fn,
            questions[idx], pretrained_model_name, generation_config
        )
        print(passage)
        samples = produce_samples(
            tokenizer, model, prompt_fn, extract_response_fn,
            questions[idx], pretrained_model_name, generation_config
        )
        print(f"len(samples): {len(samples)}")
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

    return [x.to_dict() for x in after_training_result]

def qa_evaluation_benchmark_squad(
        wandb_logger, model, tokenizer, prompt_fn, extract_response_fn, batch_size=16
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    def create_prompt(row):
        content = f"You are a helpful assistant. As an input, you will receive two texts:\n\n##QUESTION: task definition;\n\n##CONTEXT: additional information that may help answering the task.\n\nWrite the exact answer to ##QUESTION as found in ##CONTEXT, otherwise write that the answer is not available.\n\n##QUESTION: {row['question']}\n\n##CONTEXT: {row['context']}"
        messages = [
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    ds = datasets.load_dataset('rajpurkar/squad', split='validation')
    ds = ds.map(lambda x: {'formatted_prompt': create_prompt(x)})

    ids = ds['id']
    formatted_prompts = ds['formatted_prompt']
    answers = ds['answers']

    preds = []
    num_batches = ceil(len(formatted_prompts) / batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = formatted_prompts[start:end]
        tokens = tokenizer(
            batch,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)
        generated_ids = model.generate(
            **tokens, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        decoded = [extract_response_fn(x) for x in decoded]
        all_special_tokens = []
        try:
            all_special_tokens.extend(tokenizer.all_special_tokens)
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer._additional_special_tokens.items()])
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer.added_tokens_decoder.items()])
        except:
            pass
        for spectok in all_special_tokens:
            decoded = [p.replace(spectok, '') for p in decoded]
        preds.extend(decoded)

    squad_preds = [{'prediction_text': preds[idx], 'id': ids[idx]} for idx in range(len(ids))]
    squad_refs = [{'answers': x} for x in answers]

    squad_metric = evaluate.load('squad')
    squad_res = squad_metric.compute(predictions=squad_preds, references=squad_refs)
    wandb_logger.log({f'squad_{key}': val for key, val in squad_res.items()})

    oracle_questions = ds['question']
    oracle_answers = [x['text'] for x in answers]
    oracle_preds = preds

    oracle_scores_texts, oracle_scores = check_accuracy_with_chatgpt(oracle_questions, oracle_answers, oracle_preds)
    oracle_scores_texts = {k: v for k, v in enumerate(oracle_scores_texts)}
    oracle_scores = {k: v for k, v in enumerate(oracle_scores)}
    wandb_logger.log({
        'squad_oracle_acc_text': oracle_scores_texts,
        'squad_oracle_acc': oracle_scores
    })

def qa_evaluation_benchmark_sst2(
        wandb_logger, model, tokenizer, prompt_fn, extract_response_fn, batch_size=16
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    def create_prompt(row):
        content = f"You are an expert in sentiment analysis. As an input, you will receive a text:\n\n##SENTENCE: text to be analyzed.\n\nRespond with POSITIVE if ##SENTENCE contains a positive sentiment, otherwise respond with NEGATIVE if ##SENTENCE contains a negative sentiment.\n\n##SENTENCE: {row['sentence']}"
        messages = [
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    ds = datasets.load_dataset('nyu-mll/glue', subset='sst2', split='validation')
    ds = ds.map(lambda x: {'formatted_prompt': create_prompt(x)})

    formatted_prompts = ds['formatted_prompt']
    answers = ds['label']

    preds = []
    num_batches = ceil(len(formatted_prompts) / batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = formatted_prompts[start:end]
        tokens = tokenizer(
            batch,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)
        generated_ids = model.generate(
            **tokens, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        decoded = [extract_response_fn(x) for x in decoded]
        all_special_tokens = []
        try:
            all_special_tokens.extend(tokenizer.all_special_tokens)
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer._additional_special_tokens.items()])
        except:
            pass
        try:
            all_special_tokens.extend([tok.content for id, tok in tokenizer.added_tokens_decoder.items()])
        except:
            pass
        for spectok in all_special_tokens:
            decoded = [p.replace(spectok, '') for p in decoded]
        for y in decoded:
            if 'POSITIVE' in y:
                preds.append(1)
            else:
                preds.append(0)

    glue_metric = evaluate.load('glue', 'sst2')
    glue_res = glue_metric.compute(predictions=preds, references=answers)
    wandb_logger.log({f'glue_sst2_{key}': val for key, val in glue_res.items()})
